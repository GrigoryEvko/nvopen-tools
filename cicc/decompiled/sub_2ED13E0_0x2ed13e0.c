// Function: sub_2ED13E0
// Address: 0x2ed13e0
//
__int64 __fastcall sub_2ED13E0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // rbx
  __int64 *v8; // rbx
  __int64 *v9; // r14

  v6 = sub_2E5E6D0(*(_QWORD *)(a1 + 48), a2);
  if ( v6 == sub_2E5E6D0(*(_QWORD *)(a1 + 48), a3) && v6 && (*(_DWORD *)(v6 + 16) != 1 || a3 == **(_QWORD **)(v6 + 8)) )
    return 0;
  if ( !a4 )
  {
    v8 = *(__int64 **)(a3 + 64);
    v9 = &v8[*(unsigned int *)(a3 + 72)];
    if ( v8 != v9 )
    {
      while ( a2 == *v8 || (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 32), a3, *v8) )
      {
        if ( v9 == ++v8 )
          return 1;
      }
      return 0;
    }
  }
  return 1;
}
