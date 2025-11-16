// Function: sub_2EC64C0
// Address: 0x2ec64c0
//
__int64 __fastcall sub_2EC64C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rdx

  v4 = *(_QWORD *)(a1 + 56);
  v5 = *(_QWORD *)(a1 + 48);
  if ( v5 != v4 )
  {
    while ( 1 )
    {
      sub_2F8F910(v5);
      if ( *(_DWORD *)(v5 + 216) )
      {
        if ( *(_DWORD *)(v5 + 220) )
          goto LABEL_4;
LABEL_9:
        v8 = *(unsigned int *)(a3 + 8);
        if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v8 + 1, 8u, v8 + 1, v6);
          v8 = *(unsigned int *)(a3 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v5;
        v5 += 256;
        ++*(_DWORD *)(a3 + 8);
        if ( v4 == v5 )
          return sub_2F8F910(a1 + 328);
      }
      else
      {
        v7 = *(unsigned int *)(a2 + 8);
        if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v7 + 1, 8u, v7 + 1, v6);
          v7 = *(unsigned int *)(a2 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v7) = v5;
        ++*(_DWORD *)(a2 + 8);
        if ( !*(_DWORD *)(v5 + 220) )
          goto LABEL_9;
LABEL_4:
        v5 += 256;
        if ( v4 == v5 )
          return sub_2F8F910(a1 + 328);
      }
    }
  }
  return sub_2F8F910(a1 + 328);
}
