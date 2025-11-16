// Function: sub_26E1C80
// Address: 0x26e1c80
//
__int64 __fastcall sub_26E1C80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r15
  __int64 v5; // rbx
  unsigned __int8 v6; // al
  unsigned __int8 *v7; // rdx
  unsigned __int8 *v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r8
  unsigned __int8 v11; // al
  unsigned __int8 v12; // al
  __int64 v13; // rcx
  const char *v14; // rsi
  __int64 v15; // rax
  unsigned __int8 v16; // al
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // rax
  const char *v20; // rcx
  __int128 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 v25; // rdi
  __int128 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // [rsp+0h] [rbp-A0h]
  __int64 v29; // [rsp+8h] [rbp-98h]
  __int64 v30; // [rsp+10h] [rbp-90h]
  __int64 v32; // [rsp+28h] [rbp-78h] BYREF
  __int64 v33[2]; // [rsp+30h] [rbp-70h] BYREF
  char v34; // [rsp+44h] [rbp-5Ch]
  __m128i v35[5]; // [rsp+50h] [rbp-50h] BYREF

  result = *(_QWORD *)(a2 + 80);
  v28 = a2 + 72;
  v29 = result;
  if ( result != a2 + 72 )
  {
    while ( 1 )
    {
      if ( !v29 )
        BUG();
      v4 = *(_QWORD *)(v29 + 32);
      v5 = v29 + 24;
      if ( v4 != v29 + 24 )
        break;
LABEL_21:
      result = *(_QWORD *)(v29 + 8);
      v29 = result;
      if ( v28 == result )
        return result;
    }
    while ( 1 )
    {
      v7 = (unsigned __int8 *)(v4 - 24);
      if ( !v4 )
        v7 = 0;
      v8 = v7;
      v10 = sub_B10CD0((__int64)(v7 + 48));
      if ( !v10 )
        goto LABEL_8;
      if ( !unk_4F838D4 )
        break;
      v30 = v10;
      sub_3143F80(v33, v8, v9);
      if ( v34 )
      {
        v10 = v30;
        v11 = *(_BYTE *)(v30 - 16);
        if ( (v11 & 2) != 0 )
        {
          if ( *(_DWORD *)(v30 - 24) != 2 )
            goto LABEL_16;
          v24 = *(_QWORD *)(v30 - 32);
        }
        else
        {
          if ( ((*(_WORD *)(v30 - 16) >> 6) & 0xF) != 2 )
          {
LABEL_16:
            v12 = *v8;
            if ( *v8 == 85 )
              goto LABEL_46;
            goto LABEL_17;
          }
          v24 = v30 - 16 - 8LL * ((v11 >> 2) & 0xF);
        }
        if ( *(_QWORD *)(v24 + 8) )
          goto LABEL_38;
        v12 = *v8;
        if ( *v8 == 85 )
        {
LABEL_46:
          v25 = *((_QWORD *)v8 - 4);
          if ( !v25 || *(_BYTE *)v25 )
            goto LABEL_56;
          if ( *(_QWORD *)(v25 + 24) != *((_QWORD *)v8 + 10) || (*(_BYTE *)(v25 + 33) & 0x20) == 0 )
          {
LABEL_54:
            v13 = 23;
            v14 = "unknown.indirect.callee";
            if ( *(_QWORD *)(v25 + 24) == *((_QWORD *)v8 + 10) )
            {
              *(_QWORD *)&v26 = sub_BD5D20(v25);
              v14 = (const char *)sub_C16140(v26, (__int64)"selected", 8);
              v13 = v27;
            }
            goto LABEL_20;
          }
          goto LABEL_19;
        }
LABEL_17:
        if ( v12 == 34 || v12 == 40 )
        {
          v25 = *((_QWORD *)v8 - 4);
          if ( v25 && !*(_BYTE *)v25 )
            goto LABEL_54;
LABEL_56:
          v13 = 23;
          v14 = "unknown.indirect.callee";
          goto LABEL_20;
        }
LABEL_19:
        v13 = 0;
        v14 = 0;
LABEL_20:
        v35[0].m128i_i64[0] = (__int64)v14;
        v32 = LODWORD(v33[0]);
        v35[0].m128i_i64[1] = v13;
        sub_26E1BB0(a3, &v32, v35);
        v4 = *(_QWORD *)(v4 + 8);
        if ( v5 == v4 )
          goto LABEL_21;
      }
      else
      {
LABEL_8:
        v4 = *(_QWORD *)(v4 + 8);
        if ( v5 == v4 )
          goto LABEL_21;
      }
    }
    v6 = *v8;
    if ( *v8 == 85 )
    {
      v15 = *((_QWORD *)v8 - 4);
      if ( v15 && !*(_BYTE *)v15 && *(_QWORD *)(v15 + 24) == *((_QWORD *)v8 + 10) && (*(_BYTE *)(v15 + 33) & 0x20) != 0 )
        goto LABEL_8;
    }
    else if ( v6 != 34 && v6 != 40 )
    {
      goto LABEL_8;
    }
    v16 = *(_BYTE *)(v10 - 16);
    if ( (v16 & 2) != 0 )
    {
      if ( *(_DWORD *)(v10 - 24) != 2 )
        goto LABEL_28;
      v23 = *(_QWORD *)(v10 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v10 - 16) >> 6) & 0xF) != 2 )
        goto LABEL_28;
      v23 = v10 - 16 - 8LL * ((v16 >> 2) & 0xF);
    }
    if ( !*(_QWORD *)(v23 + 8) )
    {
LABEL_28:
      v33[0] = sub_C1B090(v10, unk_4F838D0);
      if ( (unsigned __int8)(*v8 - 34) > 0x33u || (v17 = 0x8000000000041LL, !_bittest64(&v17, (unsigned int)*v8 - 34)) )
        BUG();
      v18 = *((_QWORD *)v8 - 4);
      v19 = 23;
      v20 = "unknown.indirect.callee";
      if ( v18 && !*(_BYTE *)v18 && *(_QWORD *)(v18 + 24) == *((_QWORD *)v8 + 10) )
      {
        *(_QWORD *)&v21 = sub_BD5D20(v18);
        v20 = (const char *)sub_C16140(v21, (__int64)"selected", 8);
        v19 = v22;
      }
      v35[0].m128i_i64[0] = (__int64)v20;
      v35[0].m128i_i64[1] = v19;
      sub_26E1BB0(a3, v33, v35);
      goto LABEL_8;
    }
LABEL_38:
    sub_26E0290(v35[0].m128i_i64, v10);
    sub_26E1AE0(a3, v35[0].m128i_i64);
    goto LABEL_8;
  }
  return result;
}
