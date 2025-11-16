// Function: sub_2ECCC50
// Address: 0x2eccc50
//
__int64 __fastcall sub_2ECCC50(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // r15
  _QWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  _QWORD *v16; // r8
  __int64 v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rdx
  __int64 **v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  void **v29; // rax
  __int64 **v30; // rsi
  _QWORD *v31; // [rsp+8h] [rbp-98h]
  __int64 v32; // [rsp+10h] [rbp-90h] BYREF
  void **v33; // [rsp+18h] [rbp-88h]
  unsigned int v34; // [rsp+20h] [rbp-80h]
  unsigned int v35; // [rsp+24h] [rbp-7Ch]
  char v36; // [rsp+2Ch] [rbp-74h]
  _BYTE v37[16]; // [rsp+30h] [rbp-70h] BYREF
  _BYTE v38[8]; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v39; // [rsp+48h] [rbp-58h]
  int v40; // [rsp+54h] [rbp-4Ch]
  int v41; // [rsp+58h] [rbp-48h]
  char v42; // [rsp+5Ch] [rbp-44h]
  _BYTE v43[64]; // [rsp+60h] [rbp-40h] BYREF

  v31 = sub_C52410();
  v8 = sub_C959E0();
  v9 = (_QWORD *)v31[2];
  v10 = v31 + 1;
  if ( v9 )
  {
    v11 = v31 + 1;
    do
    {
      while ( 1 )
      {
        v12 = v9[2];
        v13 = v9[3];
        if ( v8 <= v9[4] )
          break;
        v9 = (_QWORD *)v9[3];
        if ( !v13 )
          goto LABEL_6;
      }
      v11 = v9;
      v9 = (_QWORD *)v9[2];
    }
    while ( v12 );
LABEL_6:
    if ( v10 != v11 && v8 >= v11[4] )
      v10 = v11;
  }
  if ( v10 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v17 = v10[7];
    v16 = v10 + 6;
    if ( v17 )
    {
      v8 = (unsigned int)dword_5020BA8;
      v18 = v10 + 6;
      do
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(v17 + 16);
          v14 = *(_QWORD *)(v17 + 24);
          if ( *(_DWORD *)(v17 + 32) >= dword_5020BA8 )
            break;
          v17 = *(_QWORD *)(v17 + 24);
          if ( !v14 )
            goto LABEL_15;
        }
        v18 = (_QWORD *)v17;
        v17 = *(_QWORD *)(v17 + 16);
      }
      while ( v15 );
LABEL_15:
      if ( v16 != v18 && dword_5020BA8 >= *((_DWORD *)v18 + 8) && *((_DWORD *)v18 + 9) )
      {
        if ( (_BYTE)qword_5020C28 )
          goto LABEL_19;
LABEL_24:
        *(_QWORD *)(a1 + 48) = 0;
        *(_QWORD *)(a1 + 8) = a1 + 32;
        *(_QWORD *)(a1 + 56) = a1 + 80;
        goto LABEL_21;
      }
    }
  }
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64, _QWORD *))(**(_QWORD **)(a3 + 16)
                                                                                                + 304LL))(
          *(_QWORD *)(a3 + 16),
          v8,
          v14,
          v15,
          v16) )
    goto LABEL_24;
LABEL_19:
  v19 = sub_2EB2140(a4, &qword_50208B0, a3);
  v20 = sub_2EB2140(a4, &qword_50209D0, a3);
  v21 = sub_BC1CD0(*(_QWORD *)(v20 + 8), &unk_4F86540, *(_QWORD *)a3);
  *(_QWORD *)(*a2 + 72LL) = a4;
  v22 = (_QWORD *)*a2;
  v23 = a2[1];
  v32 = v19 + 8;
  v33 = (void **)(v21 + 8);
  if ( !(unsigned __int8)sub_2ECC8D0(v22, a3, v23, &v32) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
LABEL_21:
    *(_QWORD *)(a1 + 64) = 2;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  sub_2EAFFB0((__int64)&v32);
  if ( v40 == v41 )
  {
    if ( v36 )
    {
      v29 = v33;
      v30 = (__int64 **)&v33[v35];
      v26 = v35;
      v25 = (__int64 **)v33;
      if ( v33 != (void **)v30 )
      {
        while ( *v25 != &qword_4F82400 )
        {
          if ( v30 == ++v25 )
          {
LABEL_30:
            while ( *v29 != &unk_4F82408 )
            {
              if ( ++v29 == (void **)v25 )
                goto LABEL_36;
            }
            goto LABEL_31;
          }
        }
        goto LABEL_31;
      }
      goto LABEL_36;
    }
    if ( sub_C8CA60((__int64)&v32, (__int64)&qword_4F82400) )
      goto LABEL_31;
  }
  if ( !v36 )
    goto LABEL_35;
  v29 = v33;
  v26 = v35;
  v25 = (__int64 **)&v33[v35];
  if ( v25 != (__int64 **)v33 )
    goto LABEL_30;
LABEL_36:
  if ( v34 > (unsigned int)v26 )
  {
    v35 = v26 + 1;
    *v25 = (__int64 *)&unk_4F82408;
    ++v32;
    goto LABEL_31;
  }
LABEL_35:
  sub_C8CC70((__int64)&v32, (__int64)&unk_4F82408, (__int64)v25, v26, v27, v28);
LABEL_31:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v37, (__int64)&v32);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v43, (__int64)v38);
  if ( !v42 )
    _libc_free(v39);
  if ( !v36 )
    _libc_free((unsigned __int64)v33);
  return a1;
}
