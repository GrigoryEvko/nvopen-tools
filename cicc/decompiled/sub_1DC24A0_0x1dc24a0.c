// Function: sub_1DC24A0
// Address: 0x1dc24a0
//
__int64 __fastcall sub_1DC24A0(__int64 *a1, __int64 a2, unsigned int a3)
{
  unsigned int v5; // esi
  __int64 v6; // r8
  unsigned int v7; // eax
  _DWORD *v8; // r13
  __int64 result; // rax
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // r15
  _WORD *v13; // rdx
  unsigned __int16 *v14; // rax
  int v15; // ebx
  unsigned __int16 *v16; // r12
  unsigned __int16 *v17; // rdx
  unsigned __int16 v18; // ax
  unsigned __int16 v19; // r9
  __int16 *v20; // r10
  __int16 *v21; // rdx
  __int16 *v22; // r11
  int v23; // edi
  __int16 v24; // di
  unsigned __int16 *v25; // r10
  unsigned int v26; // edx
  _DWORD *v27; // r12
  int v28; // eax
  unsigned __int16 *v29; // rdx
  __int16 v30; // dx
  int v31; // edx
  __int64 v32; // [rsp+4h] [rbp-40h]
  __int64 v33; // [rsp+Ch] [rbp-38h]
  __int64 v34; // [rsp+14h] [rbp-30h]

  v5 = *((_DWORD *)a1 + 4);
  v6 = a1[1];
  v34 = a1[7];
  v7 = *(unsigned __int8 *)(v34 + a3);
  if ( v7 >= v5 )
  {
LABEL_7:
    v8 = (_DWORD *)(v6 + 4LL * v5);
LABEL_8:
    if ( (*(_QWORD *)(*(_QWORD *)(a2 + 304) + 8LL * (a3 >> 6)) & (1LL << a3)) == 0 )
    {
      v10 = *a1;
      v33 = v10;
      if ( !v10 )
        BUG();
      v11 = *(_QWORD *)(v10 + 8);
      v12 = *(_QWORD *)(v10 + 56);
      v15 = a3 * (*(_DWORD *)(v11 + 24LL * a3 + 16) & 0xF);
      v13 = (_WORD *)(v12 + 2LL * (*(_DWORD *)(v11 + 24LL * a3 + 16) >> 4));
      v14 = v13 + 1;
      LOWORD(v15) = *v13 + v15;
LABEL_11:
      v16 = v14;
      while ( 1 )
      {
        if ( !v16 )
          return 1;
        v32 = *(_QWORD *)(v33 + 48);
        v17 = (unsigned __int16 *)(v32 + 4LL * (unsigned __int16)v15);
        v18 = *v17;
        v19 = v17[1];
        if ( *v17 )
          break;
LABEL_36:
        v31 = *v16;
        v14 = 0;
        ++v16;
        if ( !(_WORD)v31 )
          goto LABEL_11;
        v15 += v31;
      }
      while ( 1 )
      {
        v20 = (__int16 *)(v12 + 2LL * *(unsigned int *)(v11 + 24LL * v18 + 8));
LABEL_15:
        v21 = v20;
        v22 = v20;
        if ( v20 )
          break;
LABEL_19:
        v18 = v19;
        if ( !v19 )
          goto LABEL_36;
        v19 = 0;
      }
      while ( 1 )
      {
        v23 = v18;
        if ( v18 != a3 )
          break;
        v24 = *v21;
        v20 = 0;
        ++v21;
        if ( !v24 )
          goto LABEL_15;
        v18 += v24;
        v22 = v21;
        if ( !v21 )
          goto LABEL_19;
      }
      v25 = v16;
      while ( 1 )
      {
        v26 = *(unsigned __int8 *)(v34 + v18);
        if ( v5 > v26 )
        {
          while ( 1 )
          {
            v27 = (_DWORD *)(v6 + 4LL * v26);
            if ( v23 == *v27 )
              break;
            v26 += 256;
            if ( v5 <= v26 )
              goto LABEL_33;
          }
          if ( v27 != v8 )
            break;
        }
        do
        {
LABEL_33:
          v30 = *v22;
          if ( *v22 )
          {
            ++v22;
            v18 += v30;
          }
          else if ( v19 )
          {
            v22 = (__int16 *)(v12 + 2LL * *(unsigned int *)(v11 + 24LL * v19 + 8));
            v18 = v19;
            v19 = 0;
          }
          else
          {
            v28 = *v25;
            if ( !(_WORD)v28 )
              return 1;
            v15 += v28;
            ++v25;
            v29 = (unsigned __int16 *)(v32 + 4LL * (unsigned __int16)v15);
            v18 = *v29;
            v19 = v29[1];
            v22 = (__int16 *)(v12 + 2LL * *(unsigned int *)(v11 + 24LL * *v29 + 8));
          }
          v23 = v18;
        }
        while ( v18 == a3 );
      }
    }
    return 0;
  }
  while ( 1 )
  {
    v8 = (_DWORD *)(v6 + 4LL * v7);
    if ( *v8 == a3 )
      break;
    v7 += 256;
    if ( v5 <= v7 )
      goto LABEL_7;
  }
  result = 0;
  if ( v8 == (_DWORD *)(v6 + 4LL * v5) )
    goto LABEL_8;
  return result;
}
