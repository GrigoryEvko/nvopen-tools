// Function: sub_388F810
// Address: 0x388f810
//
__int64 __fastcall sub_388F810(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  int v3; // eax
  __int64 v4; // rax

  v3 = sub_16D1B30((__int64 *)a1, a2, a3);
  if ( v3 == -1 )
    return 0;
  v4 = *(_QWORD *)a1 + 8LL * v3;
  if ( v4 == *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) )
    return 0;
  else
    return *(_QWORD *)(*(_QWORD *)v4 + 8LL);
}
