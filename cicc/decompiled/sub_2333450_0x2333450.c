// Function: sub_2333450
// Address: 0x2333450
//
__int64 __fastcall sub_2333450(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  char v3; // r15
  char v4; // r14
  char v5; // bl
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned __int64 v9; // r8
  __int64 v10; // rdx
  char v11; // al
  bool v13; // al
  unsigned int v14; // eax
  unsigned int v15; // ebx
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // [rsp+8h] [rbp-E8h]
  __int8 v20; // [rsp+10h] [rbp-E0h]
  __int64 v21; // [rsp+10h] [rbp-E0h]
  char v22; // [rsp+18h] [rbp-D8h]
  __int64 v23; // [rsp+18h] [rbp-D8h]
  char v24; // [rsp+27h] [rbp-C9h]
  __int64 v25; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int64 v26; // [rsp+38h] [rbp-B8h]
  __int64 v27; // [rsp+48h] [rbp-A8h] BYREF
  __m128i v28; // [rsp+50h] [rbp-A0h] BYREF
  unsigned __int64 v29[4]; // [rsp+60h] [rbp-90h] BYREF
  __m128i v30; // [rsp+80h] [rbp-70h] BYREF
  _QWORD **v31; // [rsp+90h] [rbp-60h]
  __int64 v32; // [rsp+98h] [rbp-58h]
  char v33; // [rsp+A0h] [rbp-50h]
  _QWORD v34[2]; // [rsp+A8h] [rbp-48h] BYREF
  _QWORD *v35; // [rsp+B8h] [rbp-38h] BYREF

  v3 = 0;
  v4 = 0;
  v5 = 0;
  v25 = a2;
  v26 = a3;
  v22 = 0;
  v24 = 0;
  v20 = 0;
  if ( a3 )
  {
    while ( 1 )
    {
      v28 = 0u;
      v30.m128i_i8[0] = 59;
      v6 = sub_C931B0(&v25, &v30, 1u, 0);
      if ( v6 == -1 )
      {
        v8 = v25;
        v6 = v26;
        v9 = 0;
        v10 = 0;
      }
      else
      {
        v7 = v6 + 1;
        v8 = v25;
        if ( v6 + 1 > v26 )
        {
          v7 = v26;
          v9 = 0;
        }
        else
        {
          v9 = v26 - v7;
        }
        v10 = v25 + v7;
        if ( v6 > v26 )
          v6 = v26;
      }
      v28.m128i_i64[0] = v8;
      v28.m128i_i64[1] = v6;
      v25 = v10;
      v26 = v9;
      if ( v6 == 4 )
        break;
      if ( v6 == 2 )
      {
        if ( *(_WORD *)v8 == 29810 )
        {
          if ( !v5 )
            v5 = 1;
          v4 = 1;
          v3 = 0;
          goto LABEL_5;
        }
        goto LABEL_28;
      }
      if ( v6 != 8 )
      {
        if ( v6 == 6 )
        {
          if ( *(_DWORD *)v8 == 762210669 && *(_WORD *)(v8 + 4) == 29810 )
          {
            if ( !v5 )
              v5 = 1;
            v4 = 1;
            v3 = 1;
            goto LABEL_5;
          }
        }
        else if ( v6 == 12 && *(_QWORD *)v8 == 0x612D74722D6E696DLL && *(_DWORD *)(v8 + 8) == 1953656674 )
        {
          if ( !v5 )
            v5 = 1;
          v4 = 0;
          v3 = 1;
          goto LABEL_5;
        }
        goto LABEL_28;
      }
      if ( *(_QWORD *)v8 == 0x74726F62612D7472LL )
      {
        if ( v5 )
        {
          v4 = 0;
          v3 = 0;
          goto LABEL_5;
        }
        v5 = 1;
        v4 = 0;
        v3 = 0;
        if ( !v9 )
          goto LABEL_17;
      }
      else
      {
LABEL_28:
        v19 = v9;
        v13 = sub_9691B0((const void *)v28.m128i_i64[0], v28.m128i_u64[1], "merge", 5);
        v9 = v19;
        if ( v13 )
        {
          v24 = 1;
        }
        else
        {
          LOBYTE(v29[0]) = 61;
          sub_232E160(&v30, &v28, v29, 1u);
          v21 = (__int64)v31;
          v23 = v32;
          if ( !sub_9691B0((const void *)v30.m128i_i64[0], v30.m128i_u64[1], "guard", 5)
            || sub_C93CC0(v21, v23, 0, v30.m128i_i64)
            || (v20 = v30.m128i_i8[0], v30.m128i_i64[0] != v30.m128i_i8[0]) )
          {
            v14 = sub_C63BB0();
            v30.m128i_i64[1] = 44;
            v15 = v14;
            v17 = v16;
            v33 = 1;
            v30.m128i_i64[0] = (__int64)"invalid BoundsChecking pass parameter '{0}' ";
            v31 = &v35;
            v32 = 1;
            v34[0] = &unk_49DB108;
            v34[1] = &v28;
            v35 = v34;
            sub_23328D0((__int64)v29, (__int64)&v30);
            sub_23058C0(&v27, (__int64)v29, v15, v17);
            v18 = v27;
            *(_BYTE *)(a1 + 8) |= 3u;
            *(_QWORD *)a1 = v18 & 0xFFFFFFFFFFFFFFFELL;
            sub_2240A30(v29);
            return a1;
          }
          v22 = 1;
          v9 = v26;
        }
LABEL_5:
        if ( !v9 )
          goto LABEL_17;
      }
    }
    if ( *(_DWORD *)v8 == 1885434484 )
    {
      v5 = 0;
      goto LABEL_5;
    }
    goto LABEL_28;
  }
LABEL_17:
  v11 = *(_BYTE *)(a1 + 8);
  *(_BYTE *)a1 = v3;
  *(_BYTE *)(a1 + 1) = v4;
  *(_BYTE *)(a1 + 2) = v5;
  *(_BYTE *)(a1 + 8) = v11 & 0xFC | 2;
  *(_BYTE *)(a1 + 3) = v24;
  *(_BYTE *)(a1 + 4) = v20;
  *(_BYTE *)(a1 + 5) = v22;
  return a1;
}
