// Function: sub_208BEC0
// Address: 0x208bec0
//
void __fastcall sub_208BEC0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r12
  unsigned __int8 v6; // bl
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // r12
  __int64 v15; // rbx
  int v16; // r14d
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned int v19; // edx
  char *v20; // r14
  __int64 v21; // r12
  __int64 v22; // rax
  unsigned int v23; // eax
  unsigned int v24; // edx
  __int64 (*v25)(); // rax
  __int64 v26; // rax
  __int64 v27; // [rsp+0h] [rbp-60h]
  __int64 v28; // [rsp+8h] [rbp-58h]

  if ( *(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) != 20 )
  {
    sub_1E2DAF0(a2, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL) + 32LL));
    v5 = *(_QWORD *)(a2 - 24);
    v6 = *(_BYTE *)(v5 + 16);
    if ( !v6 )
    {
      if ( sub_15E4F60(*(_QWORD *)(a2 - 24)) )
      {
        v19 = *(_DWORD *)(v5 + 36);
        if ( v19
          || (v25 = *(__int64 (**)())(**(_QWORD **)(a1 + 544) + 32LL), v25 != sub_16FF770)
          && (v26 = v25()) != 0
          && (v19 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v26 + 32LL))(v26, v5)) != 0 )
        {
          v20 = sub_2081F00((__int64 *)a1, (__int64 *)a2, v19, v7, v8, v9, a3, a4, a5);
          if ( !v20 )
            return;
          v21 = *(_QWORD *)(a1 + 552);
          v22 = sub_1E0A0C0(*(_QWORD *)(v21 + 32));
          v23 = 8 * sub_15A9520(v22, 0);
          if ( v23 == 32 )
          {
            v6 = 5;
          }
          else if ( v23 > 0x20 )
          {
            if ( v23 == 64 )
            {
              v6 = 6;
            }
            else if ( v23 == 128 )
            {
              v6 = 7;
            }
          }
          else if ( v23 == 8 )
          {
            v6 = 3;
          }
          else if ( v23 == 16 )
          {
            v6 = 4;
          }
          v28 = sub_1D27640(v21, v20, v6, 0);
          v27 = v24;
          if ( *(char *)(a2 + 23) >= 0 )
            goto LABEL_23;
          goto LABEL_6;
        }
      }
      v5 = *(_QWORD *)(a2 - 24);
    }
    v28 = (__int64)sub_20685E0(a1, (__int64 *)v5, a3, a4, a5);
    v27 = v10;
    if ( *(char *)(a2 + 23) >= 0 )
    {
LABEL_23:
      sub_20789D0(a1, a2 | 4, v28, v27, (*(_WORD *)(a2 + 18) & 3u) - 1 <= 1, 0, a3, a4, a5);
      return;
    }
LABEL_6:
    v11 = sub_1648A40(a2);
    v13 = v11 + v12;
    if ( *(char *)(a2 + 23) < 0 )
      v13 -= sub_1648A40(a2);
    v14 = v13 >> 4;
    if ( (_DWORD)v14 )
    {
      v15 = 0;
      v16 = 0;
      v17 = 16LL * (unsigned int)v14;
      do
      {
        v18 = 0;
        if ( *(char *)(a2 + 23) < 0 )
          v18 = sub_1648A40(a2);
        v16 += *(_DWORD *)(*(_QWORD *)(v18 + v15) + 8LL) == 0;
        v15 += 16;
      }
      while ( v15 != v17 );
      if ( v16 )
      {
        sub_20A06E0(a1, a2 | 4, v28, v27, 0);
        return;
      }
    }
    goto LABEL_23;
  }
  sub_2079C70(a1, a2 | 4, a3, a4, a5);
}
