// Function: sub_14134C0
// Address: 0x14134c0
//
__int64 __fastcall sub_14134C0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax

  result = *a2;
  if ( *(_BYTE *)(*a2 + 8) == 15 )
  {
    sub_1412F90(a1, (unsigned __int64)a2 & 0xFFFFFFFFFFFFFFFBLL);
    sub_1412F90(a1, (unsigned __int64)a2 & 0xFFFFFFFFFFFFFFFBLL | 4);
    return sub_143DD10(*(_QWORD *)(a1 + 288), a2);
  }
  return result;
}
