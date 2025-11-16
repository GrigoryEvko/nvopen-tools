// Function: sub_37F5D50
// Address: 0x37f5d50
//
void __fastcall sub_37F5D50(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  _BYTE *v23; // rdi
  __int16 *v24; // rsi
  int v25; // edx
  unsigned __int64 *v26; // rdx
  __int64 v27; // rcx
  unsigned __int64 v28; // rsi
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // r14
  __int64 v34; // rsi
  __int64 *v35; // [rsp+0h] [rbp-90h]
  __int64 v36; // [rsp+10h] [rbp-80h] BYREF
  _BYTE *v37; // [rsp+18h] [rbp-78h]
  __int64 v38; // [rsp+20h] [rbp-70h]
  _BYTE v39[48]; // [rsp+28h] [rbp-68h] BYREF
  int v40; // [rsp+58h] [rbp-38h]

  if ( !*(_BYTE *)(a5 + 28) )
  {
    if ( sub_C8CA60(a5, a2) )
      return;
    if ( !*(_BYTE *)(a5 + 28) )
      goto LABEL_25;
    v11 = *(__int64 **)(a5 + 8);
    v13 = *(unsigned int *)(a5 + 20);
    v14 = &v11[v13];
    if ( v14 != v11 )
    {
LABEL_8:
      while ( a2 != *v11 )
      {
        if ( ++v11 == v14 )
          goto LABEL_32;
      }
LABEL_9:
      v15 = *(_QWORD *)(a1 + 208);
      v36 = 0;
      v37 = v39;
      v38 = 0x600000000LL;
      v40 = 0;
      sub_2ED4FB0((__int64)&v36, v15, (__int64)v11, v12, a5, a6);
      sub_2E225E0(&v36, a2, v16, v17, v18, v19);
      if ( a3 - 1 <= 0x3FFFFFFE )
      {
        v23 = v37;
        v22 = 1;
        v20 = *(_DWORD *)(*(_QWORD *)(v36 + 8) + 24LL * a3 + 16) & 0xFFF;
        v24 = (__int16 *)(*(_QWORD *)(v36 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v36 + 8) + 24LL * a3 + 16) >> 12));
        do
        {
          if ( !v24 )
            break;
          if ( (*(_QWORD *)&v37[8 * ((unsigned int)v20 >> 6)] & (1LL << v20)) != 0 )
            goto LABEL_16;
          v25 = *v24++;
          v20 = (unsigned int)(v25 + v20);
        }
        while ( (_WORD)v25 );
        goto LABEL_14;
      }
LABEL_16:
      v28 = sub_37F5B00(a1, a2, a3, v20, v21, v22);
      if ( v28 )
      {
        if ( !*(_BYTE *)(a4 + 28) )
        {
LABEL_27:
          sub_C8CC70(a4, v28, (__int64)v26, v27, v29, v30);
          v23 = v37;
LABEL_14:
          if ( v23 != v39 )
            _libc_free((unsigned __int64)v23);
          return;
        }
        v31 = *(unsigned __int64 **)(a4 + 8);
        v27 = *(unsigned int *)(a4 + 20);
        v26 = &v31[v27];
        if ( v31 == v26 )
        {
LABEL_26:
          if ( (unsigned int)v27 >= *(_DWORD *)(a4 + 16) )
            goto LABEL_27;
          *(_DWORD *)(a4 + 20) = v27 + 1;
          *v26 = v28;
          ++*(_QWORD *)a4;
        }
        else
        {
          while ( v28 != *v31 )
          {
            if ( v26 == ++v31 )
              goto LABEL_26;
          }
        }
      }
      else
      {
        v32 = *(unsigned int *)(a2 + 72);
        v33 = *(__int64 **)(a2 + 64);
        v35 = &v33[v32];
        if ( v33 != v35 )
        {
          do
          {
            v34 = *v33++;
            sub_37F5D50(a1, v34, a3, a4, a5);
          }
          while ( v35 != v33 );
          v23 = v37;
          goto LABEL_14;
        }
      }
      v23 = v37;
      goto LABEL_14;
    }
LABEL_32:
    if ( *(_DWORD *)(a5 + 16) > (unsigned int)v13 )
    {
      *(_DWORD *)(a5 + 20) = v13 + 1;
      *v14 = a2;
      ++*(_QWORD *)a5;
      goto LABEL_9;
    }
LABEL_25:
    sub_C8CC70(a5, a2, (__int64)v11, v12, a5, a6);
    goto LABEL_9;
  }
  v11 = *(__int64 **)(a5 + 8);
  v12 = (__int64)&v11[*(unsigned int *)(a5 + 20)];
  LODWORD(v13) = *(_DWORD *)(a5 + 20);
  v14 = v11;
  if ( v11 == (__int64 *)v12 )
    goto LABEL_32;
  while ( a2 != *v14 )
  {
    if ( (__int64 *)v12 == ++v14 )
      goto LABEL_8;
  }
}
