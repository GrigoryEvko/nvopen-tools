// Function: sub_17D9C10
// Address: 0x17d9c10
//
unsigned __int64 __fastcall sub_17D9C10(_QWORD *a1, __int64 a2)
{
  char v4; // cl
  const char **v5; // rbx
  __int64 v6; // rax
  const char **v7; // rdx
  __int64 v8; // rax
  const char *v9; // r14
  __int128 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r14
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdi
  __int64 v24; // rsi
  unsigned __int64 result; // rax
  __int64 v26; // [rsp+8h] [rbp-B8h]
  __int64 v27; // [rsp+18h] [rbp-A8h]
  __int64 v28[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v29; // [rsp+30h] [rbp-90h]
  __int64 v30[16]; // [rsp+40h] [rbp-80h] BYREF

  sub_17CE510((__int64)v30, a2, 0, 0, 0);
  v4 = *(_BYTE *)(a2 + 23);
  if ( (v4 & 0x40) != 0 )
  {
    v5 = *(const char ***)(a2 - 8);
    v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  }
  else
  {
    v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v5 = (const char **)(a2 - 24 * v6);
  }
  v27 = 0;
  while ( 1 )
  {
    v7 = (const char **)a2;
    v8 = 24 * v6;
    if ( (v4 & 0x40) != 0 )
      v7 = (const char **)(*(_QWORD *)(a2 - 8) + v8);
    if ( v5 == v7 )
      break;
    v9 = *v5;
    *(_QWORD *)&v10 = a1;
    *((_QWORD *)&v10 + 1) = *v5;
    v13 = (__int64)sub_17D4DA0(v10);
    if ( *(_DWORD *)(a1[1] + 156LL) )
    {
      v14 = (__int64)v9;
      v17 = sub_17D4880((__int64)a1, v9, v11, v12);
      if ( *(_DWORD *)(a1[1] + 156LL) )
      {
        if ( v27 )
        {
          if ( *(_BYTE *)(v17 + 16) > 0x10u || !sub_1593BB0(v17, v14, v15, v16) )
          {
            if ( *(_BYTE *)(*(_QWORD *)v13 + 8LL) == 16 )
            {
              v26 = *(_QWORD *)v13;
              v18 = sub_1643030(*(_QWORD *)(*(_QWORD *)v13 + 24LL));
              v19 = sub_1644900(*(_QWORD **)(a1[1] + 168LL), *(_DWORD *)(v26 + 32) * v18);
              if ( v26 != v19 )
              {
                v29 = 257;
                v13 = sub_12AA3B0(v30, 0x2Fu, v13, v19, (__int64)v28);
              }
            }
            v29 = 257;
            v20 = *(_QWORD *)v13;
            v23 = (__int64)sub_17CD8D0(a1, *(_QWORD *)v13);
            if ( v23 )
              v23 = sub_15A06D0((__int64 **)v23, v20, v21, v22);
            if ( *(_BYTE *)(v13 + 16) > 0x10u || *(_BYTE *)(v23 + 16) > 0x10u )
              v24 = (__int64)sub_17CD110(v30, 33, v13, v23, v28);
            else
              v24 = sub_15A37B0(0x21u, (_QWORD *)v13, (_QWORD *)v23, 0);
            v29 = 257;
            v27 = sub_156B790(v30, v24, v17, v27, (__int64)v28, 0);
          }
        }
        else
        {
          v27 = v17;
        }
      }
    }
    v4 = *(_BYTE *)(a2 + 23);
    v5 += 3;
    v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  }
  result = *(unsigned int *)(a1[1] + 156LL);
  if ( (_DWORD)result )
    result = sub_17D4B80((__int64)a1, a2, v27);
  if ( v30[0] )
    return sub_161E7C0((__int64)v30, v30[0]);
  return result;
}
