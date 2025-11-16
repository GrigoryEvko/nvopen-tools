// Function: sub_1F3D090
// Address: 0x1f3d090
//
__int64 __fastcall sub_1F3D090(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(*(_QWORD *)a1 + 216LL);
  if ( v1 == sub_1F3CA20 )
    return 0;
  else
    return v1();
}
