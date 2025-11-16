// Function: sub_2042530
// Address: 0x2042530
//
__int64 __fastcall sub_2042530(_QWORD *a1, __int64 *a2, char a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 v6; // rax
  int v8; // eax
  int v9; // edx
  __int64 v10; // r14
  int v11; // [rsp+4h] [rbp-3Ch]
  int v12; // [rsp+4h] [rbp-3Ch]
  __int64 *v13; // [rsp+8h] [rbp-38h]

  if ( a2 )
  {
    v3 = 0;
    if ( *a2 && *(__int16 *)(*a2 + 24) < 0 )
    {
      v4 = a1[16];
      v5 = *(__int64 **)(v4 + 256);
      v13 = *(__int64 **)(v4 + 264);
      if ( a3 )
      {
        if ( *(__int64 **)(v4 + 264) != v5 )
        {
          do
          {
            v6 = *v5++;
            v3 += sub_20422C0((__int64)a1, a2, *(unsigned __int16 *)(*(_QWORD *)v6 + 24LL));
          }
          while ( v13 != v5 );
        }
      }
      else
      {
        while ( v13 != v5 )
        {
          v10 = *v5;
          v12 = *(_DWORD *)(a1[9] + 4LL * *(unsigned __int16 *)(*(_QWORD *)*v5 + 24LL));
          if ( (unsigned int)sub_20422C0((__int64)a1, a2, *(unsigned __int16 *)(*(_QWORD *)*v5 + 24LL)) + v12 )
          {
            v11 = *(_DWORD *)(a1[9] + 4LL * *(unsigned __int16 *)(*(_QWORD *)v10 + 24LL));
            v8 = sub_20422C0((__int64)a1, a2, *(unsigned __int16 *)(*(_QWORD *)v10 + 24LL));
            v9 = *(unsigned __int16 *)(*(_QWORD *)v10 + 24LL);
            if ( (unsigned int)(v8 + v11) >= *(_DWORD *)(a1[12] + 4LL * (unsigned __int16)v9) )
              v3 += sub_20422C0((__int64)a1, a2, v9);
          }
          ++v5;
        }
      }
    }
  }
  else
  {
    return 0;
  }
  return v3;
}
