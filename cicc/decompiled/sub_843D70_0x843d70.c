// Function: sub_843D70
// Address: 0x843d70
//
__int64 __fastcall sub_843D70(__m128i *a1, __int64 a2, _BYTE *a3, unsigned int a4)
{
  __int64 v4; // r8
  __int64 v8; // r15
  char v9; // al
  int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 result; // rax
  int v16; // eax
  int v17; // eax
  __int64 v18; // rdi
  char v19; // al
  int v20; // eax
  __int64 v21; // rcx
  __int64 k; // rax
  int v23; // eax
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // r10
  __int64 v32; // rdi
  int v33; // eax
  __int64 i; // rax
  int v35; // eax
  __int64 j; // rax
  __int64 v37; // [rsp-10h] [rbp-1D0h]
  __int64 v38; // [rsp-8h] [rbp-1C8h]
  __int64 v39; // [rsp+10h] [rbp-1B0h]
  __int64 v40; // [rsp+10h] [rbp-1B0h]
  __int64 v41; // [rsp+10h] [rbp-1B0h]
  __int64 v42; // [rsp+18h] [rbp-1A8h]
  __int64 v43; // [rsp+18h] [rbp-1A8h]
  __int64 v44; // [rsp+18h] [rbp-1A8h]
  __int64 v45; // [rsp+18h] [rbp-1A8h]
  _BOOL4 v46; // [rsp+2Ch] [rbp-194h] BYREF
  __m128i v47[25]; // [rsp+30h] [rbp-190h] BYREF

  v4 = a2;
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(_BYTE *)(a2 + 34);
  v10 = (2 * v9) & 0x40;
  v46 = (v9 & 4) != 0;
  if ( dword_4F077C4 == 2 )
  {
    v24 = sub_8D23B0(v8);
    v4 = a2;
    if ( v24 )
    {
      sub_8AE000(v8);
      v4 = a2;
    }
  }
  if ( (*(_BYTE *)(v4 + 32) & 1) == 0 || a3 && (a3[16] & 2) != 0 )
  {
    if ( dword_4F077BC && qword_4F077A8 > 0x76BFu )
    {
      v42 = v4;
      v17 = sub_8D2FB0(v8);
      v4 = v42;
      if ( v17
        && a1[1].m128i_i16[0] == 257
        && (v25 = a1[9].m128i_i64[0], *(_BYTE *)(v25 + 24) == 1)
        && (unsigned __int8)(*(_BYTE *)(v25 + 56) - 94) <= 1u
        && ((v26 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v25 + 72) + 16LL) + 56LL), (*(_BYTE *)(v26 + 144) & 1) != 0)
         || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v26 + 40) + 32LL) + 179LL) & 0x20) != 0) )
      {
        v27 = sub_8D46C0(v8);
        v4 = v42;
        if ( (*(_BYTE *)(v27 + 140) & 0xFB) == 8 )
        {
          v39 = v27;
          v28 = sub_8D4C10(v27, dword_4F077C4 != 2);
          v4 = v42;
          if ( (v28 & 1) != 0 )
          {
            v30 = a1->m128i_i64[0];
            v31 = v39;
            if ( a1->m128i_i64[0] == v39
              || (v32 = v39, v40 = v42, v44 = v31, v33 = sub_8D97D0(v32, v30, 32, v29, v4), v31 = v44, v4 = v40, v33) )
            {
              for ( i = v31; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
                ;
              if ( *(_DWORD *)(i + 136) != 1 )
              {
                v41 = v4;
                v45 = v31;
                v35 = sub_8D3A70(v31);
                v4 = v41;
                if ( !v35 )
                  goto LABEL_44;
                for ( j = v45; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                  ;
                if ( *(char *)(*(_QWORD *)(*(_QWORD *)j + 96LL) + 178LL) < 0 )
                {
LABEL_44:
                  sub_68F8E0(v47, a1);
                  sub_8283A0((__int64)a1, v45, 1, 0);
                  sub_6E4BC0((__int64)a1, (__int64)v47);
                  v4 = v41;
                }
              }
            }
          }
        }
      }
      else if ( dword_4F077BC )
      {
        if ( qword_4F077A8 > 0x9D07u )
        {
          v18 = a1->m128i_i64[0];
          if ( (*(_BYTE *)(a1->m128i_i64[0] + 140) & 0xFB) == 8 )
          {
            v19 = sub_8D4C10(v18, dword_4F077C4 != 2);
            v4 = v42;
            if ( (v19 & 2) != 0 && a1[1].m128i_i8[1] == 1 )
            {
              v20 = sub_8D3A70(v8);
              v4 = v42;
              if ( v20 )
              {
                for ( k = v8; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                  ;
                if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)k + 96LL) + 177LL) & 0x40) != 0 )
                {
                  if ( a1->m128i_i64[0] == v8 || (v23 = sub_8D97D0(v8, a1->m128i_i64[0], 32, v21, v42), v4 = v42, v23) )
                  {
                    v43 = v4;
                    sub_6F7690(a1, v8);
                    v4 = v43;
                  }
                }
              }
            }
          }
        }
      }
    }
    v16 = v10;
    if ( (*(_BYTE *)(v4 + 34) & 0x40) != 0 )
    {
      BYTE1(v16) = BYTE1(v10) | 8;
      v10 = v16;
    }
    sub_843C40(a1, v8, (__int64)&v46, a3, 1, v10 | 0x10000000, a4);
    v11 = v37;
    v12 = v38;
    result = dword_4A52070[0];
    if ( dword_4A52070[0] )
      return sub_6E6B60(a1, 0, v11, v12, v13, v14);
  }
  else
  {
    sub_847420(a1, v8, a3, a4);
    result = dword_4A52070[0];
    if ( dword_4A52070[0] )
      return sub_6E6B60(a1, 0, v11, v12, v13, v14);
  }
  return result;
}
