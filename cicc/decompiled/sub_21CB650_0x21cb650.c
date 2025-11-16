// Function: sub_21CB650
// Address: 0x21cb650
//
_QWORD *__fastcall sub_21CB650(_QWORD *a1, __int64 *a2, __int64 *a3, unsigned int a4, char a5, int a6)
{
  _QWORD *v7; // r15
  _DWORD *v9; // rsi
  unsigned __int64 v10; // r13
  _DWORD *v11; // rax
  _QWORD *v12; // r8
  unsigned int v13; // ebx
  signed int v14; // r9d
  unsigned int *v15; // rsi
  unsigned int v16; // r12d
  __int64 v17; // r11
  __int64 v18; // rax
  __int64 v19; // rbx
  unsigned int v20; // r10d
  __int64 v21; // r15
  __int64 v22; // r14
  __int64 v23; // r13
  int v24; // eax
  char v25; // cl
  int v26; // edi
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // r13
  unsigned int v31; // r13d
  int v32; // eax
  __int64 v33; // rax
  unsigned int v34; // [rsp+4h] [rbp-9Ch]
  _QWORD *v35; // [rsp+8h] [rbp-98h]
  __int64 v36; // [rsp+18h] [rbp-88h]
  signed int v37; // [rsp+20h] [rbp-80h]
  char v38; // [rsp+28h] [rbp-78h]
  __int64 v41; // [rsp+48h] [rbp-58h]
  __int64 v42; // [rsp+50h] [rbp-50h]
  int v43; // [rsp+58h] [rbp-48h]
  __m128i v45; // [rsp+60h] [rbp-40h] BYREF

  v7 = a1;
  v9 = a1 + 2;
  *a1 = a1 + 2;
  a1[1] = 0x1000000000LL;
  v10 = *((unsigned int *)a2 + 2);
  if ( (unsigned int)v10 > 0x10 )
  {
    sub_16CD150((__int64)a1, v9, v10, 4, a5, a6);
    v9 = (_DWORD *)*a1;
  }
  v11 = &v9[v10];
  for ( *((_DWORD *)a1 + 2) = v10; v11 != v9; ++v9 )
    *v9 = 3;
  if ( a5 )
    return v7;
  v43 = *((_DWORD *)a2 + 2);
  if ( !v43 )
    return v7;
  v12 = a1;
  v13 = 0;
LABEL_8:
  while ( 2 )
  {
    v14 = v13;
    v15 = (unsigned int *)&unk_435DB90;
    v16 = 16;
    v17 = 16LL * v13;
    v18 = v13 + 1;
    v19 = 8LL * v13;
    v42 = 8 * v18;
    v20 = v18;
    v41 = 16 * v18;
    while ( 1 )
    {
      if ( v16 > a4 )
        goto LABEL_29;
      v21 = *a3;
      v22 = *(_QWORD *)(*a3 + v19) & (v16 - 1);
      if ( v22 )
        goto LABEL_29;
      v34 = v20;
      v35 = v12;
      v23 = *a2;
      v36 = v17;
      v37 = v14;
      v38 = *(_BYTE *)(*a2 + v17);
      v45 = _mm_loadu_si128((const __m128i *)(*a2 + v17));
      if ( v38 )
      {
        v24 = sub_1F3E310(&v45);
        v25 = v38;
        v14 = v37;
        v17 = v36;
        v26 = v24;
        v12 = v35;
        v20 = v34;
      }
      else
      {
        v32 = sub_1F58D40((__int64)&v45);
        v20 = v34;
        v14 = v37;
        v12 = v35;
        v26 = v32;
        v17 = v36;
        v25 = 0;
      }
      v27 = (unsigned int)(v26 + 7) >> 3;
      if ( v16 <= (unsigned int)v27 )
        goto LABEL_29;
      v28 = v16 / (unsigned int)v27;
      if ( v16 != v16 / (unsigned int)v27 * (_DWORD)v27
        || v14 + v28 > *((_DWORD *)a2 + 2)
        || ((v28 - 2) & 0xFFFFFFFD) != 0 )
      {
        goto LABEL_29;
      }
      if ( v14 + v28 > v20 )
      {
        v29 = v41 + v23;
        while ( *(_BYTE *)v29 == v25
             && (v25 || v45.m128i_i64[1] == *(_QWORD *)(v29 + 8))
             && *(_QWORD *)(v21 + v42 + 8 * v22) - *(_QWORD *)(v21 + 8LL * (unsigned int)(v14 + v22)) == v27 )
        {
          ++v22;
          v29 += 16;
          if ( v28 - 1 == v22 )
            goto LABEL_23;
        }
        goto LABEL_29;
      }
LABEL_23:
      if ( v28 == 4 )
        break;
      if ( v28 != 1 )
      {
        *(_DWORD *)(*v12 + 4LL * v14) = 1;
        *(_DWORD *)(*v12 + 4LL * v14 + 4) = 2;
        v13 = v20 + 1;
        if ( v43 != v20 + 1 )
          goto LABEL_8;
        return v12;
      }
LABEL_29:
      if ( &unk_435DBA0 == (_UNKNOWN *)++v15 )
      {
        v31 = v20;
        goto LABEL_31;
      }
      v16 = *v15;
    }
    v33 = 4LL * v14;
    *(_DWORD *)(*v12 + v33) = 1;
    v31 = v14 + 4;
    *(_DWORD *)(*v12 + v33 + 4) = 0;
    *(_DWORD *)(*v12 + v33 + 8) = 0;
    *(_DWORD *)(*v12 + v33 + 12) = 2;
LABEL_31:
    v13 = v31;
    if ( v43 != v31 )
      continue;
    return v12;
  }
}
