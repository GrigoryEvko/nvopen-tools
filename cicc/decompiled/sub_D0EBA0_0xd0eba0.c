// Function: sub_D0EBA0
// Address: 0xd0eba0
//
__int64 __fastcall sub_D0EBA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r12
  __int64 v9; // rsi
  int v11; // ecx
  __int64 v12; // rsi
  int v13; // ecx
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r9
  unsigned int v17; // r13d
  __int64 v18; // rsi
  unsigned int v19; // eax
  __int64 v20; // r9
  _QWORD *v21; // r12
  unsigned __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r12
  int v25; // eax
  unsigned __int64 v26; // rdi
  int v27; // eax
  __int64 *v28; // rdx
  unsigned int i; // r13d
  __int64 v30; // rax
  int v31; // eax
  int v32; // eax
  int v33; // r13d
  int v34; // [rsp+8h] [rbp-168h]
  int v35; // [rsp+1Ch] [rbp-154h]
  __int64 v36; // [rsp+20h] [rbp-150h]
  __int64 *v38; // [rsp+28h] [rbp-148h]
  _BYTE *v39; // [rsp+30h] [rbp-140h] BYREF
  __int64 v40; // [rsp+38h] [rbp-138h]
  _BYTE v41[304]; // [rsp+40h] [rbp-130h] BYREF

  v8 = *(_QWORD *)(a1 + 40);
  v9 = *(_QWORD *)(a2 + 40);
  if ( v8 != v9 )
    return sub_D0E9D0(*(_QWORD *)(a1 + 40), v9, a3, a4, a5);
  if ( !a5 || (v11 = *(_DWORD *)(a5 + 24), v12 = *(_QWORD *)(a5 + 8), !v11) )
  {
LABEL_9:
    if ( a1 == a2 )
      return 1;
    v18 = a2;
    LOBYTE(v19) = sub_B445A0(a1, a2);
    v17 = v19;
    if ( (_BYTE)v19 )
      return 1;
    if ( sub_AA5B70(v8) )
      return v17;
    v21 = (_QWORD *)(v8 + 48);
    v39 = v41;
    v40 = 0x2000000000LL;
    v22 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v22 == v21 )
      return v17;
    v23 = a5;
    if ( !v22 )
      BUG();
    v24 = v22 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v22 - 24) - 30 > 0xA )
    {
      v34 = 0;
      v27 = 0;
    }
    else
    {
      v25 = sub_B46E30(v24);
      v18 = (__int64)v41;
      v23 = a5;
      v26 = v25;
      v35 = v25;
      v27 = 0;
      v34 = v26;
      v28 = (__int64 *)v41;
      if ( v26 > 0x20 )
      {
        sub_C8D5F0((__int64)&v39, v41, v26, 8u, a5, v20);
        v27 = v40;
        v23 = a5;
        v28 = (__int64 *)&v39[8 * (unsigned int)v40];
      }
      if ( v35 )
      {
        for ( i = 0; i != v35; ++i )
        {
          if ( v28 )
          {
            v18 = i;
            v36 = v23;
            v38 = v28;
            v30 = sub_B46EC0(v24, i);
            v28 = v38;
            v23 = v36;
            *v38 = v30;
          }
          ++v28;
        }
        v31 = v40 + v26;
        goto LABEL_23;
      }
    }
    v31 = v34 + v27;
LABEL_23:
    LODWORD(v40) = v31;
    v17 = 0;
    if ( v31 )
    {
      v18 = *(_QWORD *)(a2 + 40);
      v17 = sub_D0E9A0((__int64)&v39, v18, a3, a4, v23, v20);
    }
    if ( v39 != v41 )
      _libc_free(v39, v18);
    return v17;
  }
  v13 = v11 - 1;
  v14 = v13 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v15 = (__int64 *)(v12 + 16LL * v14);
  v16 = *v15;
  if ( v8 != *v15 )
  {
    v32 = 1;
    while ( v16 != -4096 )
    {
      v33 = v32 + 1;
      v14 = v13 & (v32 + v14);
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v8 == *v15 )
        goto LABEL_6;
      v32 = v33;
    }
    goto LABEL_9;
  }
LABEL_6:
  if ( !v15[1] )
    goto LABEL_9;
  return 1;
}
