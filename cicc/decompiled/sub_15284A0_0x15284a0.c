// Function: sub_15284A0
// Address: 0x15284a0
//
void __fastcall sub_15284A0(_DWORD **a1, __int64 a2, unsigned int a3)
{
  unsigned __int64 v3; // r15
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  char v12; // dl
  __int64 v13; // rax
  __int64 v14; // r8
  unsigned __int64 v15; // rbx
  __int64 **v16; // r15
  int v17; // r9d
  __int64 v18; // rax
  __int64 **v19; // rbx
  __int64 *v20; // rsi
  __int64 v21; // [rsp+8h] [rbp-168h]
  int v22; // [rsp+10h] [rbp-160h]
  char v23; // [rsp+17h] [rbp-159h]
  __int64 v24; // [rsp+18h] [rbp-158h]
  unsigned __int64 v25; // [rsp+20h] [rbp-150h]
  __int64 v26; // [rsp+28h] [rbp-148h]
  _BYTE *v27; // [rsp+30h] [rbp-140h] BYREF
  __int64 v28; // [rsp+38h] [rbp-138h]
  _BYTE v29[304]; // [rsp+40h] [rbp-130h] BYREF

  v3 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v27 = v29;
  v28 = 0x4000000000LL;
  v25 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v24 = sub_16498A0(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v23 = (a2 >> 2) & 1;
  if ( !v23 )
    v3 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(char *)(v3 + 23) < 0 )
  {
    v6 = sub_1648A40(v3);
    v8 = v6 + v7;
    if ( *(char *)(v3 + 23) >= 0 )
      v9 = v8 >> 4;
    else
      LODWORD(v9) = (v8 - sub_1648A40(v25)) >> 4;
    if ( (_DWORD)v9 )
    {
      v26 = 0;
      v21 = 16LL * (unsigned int)v9;
      while ( 1 )
      {
        v10 = v25;
        v11 = 0;
        v12 = *(_BYTE *)(v25 + 23);
        if ( !v23 )
          break;
        if ( v12 < 0 )
          goto LABEL_10;
LABEL_11:
        v13 = v26 + v11;
        v14 = 24LL * *(unsigned int *)(v13 + 8);
        v15 = 0xAAAAAAAAAAAAAAABLL * ((24LL * *(unsigned int *)(v13 + 12) - v14) >> 3);
        v16 = (__int64 **)(v25 + v14 - 24LL * (*(_DWORD *)(v25 + 20) & 0xFFFFFFF));
        v17 = sub_16032C0(v24, *(_QWORD *)v13 + 16LL, **(_QWORD **)v13);
        v18 = (unsigned int)v28;
        if ( (unsigned int)v28 >= HIDWORD(v28) )
        {
          v22 = v17;
          sub_16CD150(&v27, v29, 0, 4);
          v18 = (unsigned int)v28;
          v17 = v22;
        }
        *(_DWORD *)&v27[4 * v18] = v17;
        v19 = &v16[3 * v15];
        LODWORD(v28) = v28 + 1;
        while ( v19 != v16 )
        {
          v20 = *v16;
          v16 += 3;
          sub_1525BE0((__int64)a1, v20, a3, (__int64)&v27);
        }
        sub_1528260(*a1, 0x37u, (__int64)&v27, 0);
        v26 += 16;
        LODWORD(v28) = 0;
        if ( v21 == v26 )
          goto LABEL_16;
      }
      if ( v12 >= 0 )
        goto LABEL_11;
      v10 = v25;
LABEL_10:
      v11 = sub_1648A40(v10);
      goto LABEL_11;
    }
  }
LABEL_16:
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
}
