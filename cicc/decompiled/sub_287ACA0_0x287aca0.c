// Function: sub_287ACA0
// Address: 0x287aca0
//
__int64 __fastcall sub_287ACA0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, _QWORD *a5)
{
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 **v18; // rdx
  __int64 v19; // rcx
  void **v20; // rax
  int v21; // eax
  __int64 *v22; // rax
  void **v23; // rax
  __int64 **v24; // rsi
  unsigned __int64 v25; // [rsp+8h] [rbp-B8h]
  __int64 *v26; // [rsp+10h] [rbp-B0h]
  __int64 v27; // [rsp+18h] [rbp-A8h]
  __int64 v28; // [rsp+20h] [rbp-A0h]
  __int64 v29; // [rsp+28h] [rbp-98h]
  __int64 v30; // [rsp+30h] [rbp-90h] BYREF
  void **v31; // [rsp+38h] [rbp-88h]
  unsigned int v32; // [rsp+40h] [rbp-80h]
  unsigned int v33; // [rsp+44h] [rbp-7Ch]
  char v34; // [rsp+4Ch] [rbp-74h]
  _BYTE v35[16]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v36; // [rsp+60h] [rbp-60h] BYREF
  void **v37; // [rsp+68h] [rbp-58h]
  unsigned int v38; // [rsp+74h] [rbp-4Ch]
  int v39; // [rsp+78h] [rbp-48h]
  char v40; // [rsp+7Ch] [rbp-44h]
  _BYTE v41[64]; // [rsp+80h] [rbp-40h] BYREF

  v25 = a5[9];
  v7 = a5[4];
  v8 = a5[2];
  v26 = (__int64 *)a5[5];
  v27 = a5[6];
  v28 = a5[3];
  v29 = a5[1];
  v10 = sub_22D3D20(a4, (__int64 *)&unk_4FDB6B0, a3, (__int64)a5) + 8;
  if ( !(unsigned __int8)sub_2877B80((__int64)a3, v10, v7, v8, v28, v27, v29, v26, v25) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  sub_22D0390((__int64)&v30, v10, v11, v12, v13, v14);
  if ( a5[9] )
  {
    if ( v40 )
    {
      v18 = (__int64 **)&v37[v38];
      v19 = v38;
      if ( v37 == (void **)v18 )
      {
LABEL_25:
        v21 = v39;
      }
      else
      {
        v20 = v37;
        while ( *v20 != &unk_4F8F810 )
        {
          if ( v18 == (__int64 **)++v20 )
            goto LABEL_25;
        }
        v18 = (__int64 **)v37[--v38];
        *v20 = v18;
        v19 = v38;
        ++v36;
        v21 = v39;
      }
    }
    else
    {
      v22 = sub_C8CA60((__int64)&v36, (__int64)&unk_4F8F810);
      if ( v22 )
      {
        *v22 = -2;
        ++v36;
        v19 = v38;
        v21 = ++v39;
      }
      else
      {
        v19 = v38;
        v21 = v39;
      }
    }
    if ( (_DWORD)v19 == v21 )
    {
      if ( v34 )
      {
        v23 = v31;
        v24 = (__int64 **)&v31[v33];
        v19 = v33;
        v18 = (__int64 **)v31;
        if ( v31 != (void **)v24 )
        {
          while ( *v18 != &qword_4F82400 )
          {
            if ( v24 == ++v18 )
            {
LABEL_20:
              while ( *v23 != &unk_4F8F810 )
              {
                if ( ++v23 == (void **)v18 )
                  goto LABEL_22;
              }
              goto LABEL_11;
            }
          }
          goto LABEL_11;
        }
        goto LABEL_22;
      }
      if ( sub_C8CA60((__int64)&v30, (__int64)&qword_4F82400) )
        goto LABEL_11;
    }
    if ( !v34 )
      goto LABEL_24;
    v23 = v31;
    v19 = v33;
    v18 = (__int64 **)&v31[v33];
    if ( v18 != (__int64 **)v31 )
      goto LABEL_20;
LABEL_22:
    if ( (unsigned int)v19 < v32 )
    {
      v33 = v19 + 1;
      *v18 = (__int64 *)&unk_4F8F810;
      ++v30;
      goto LABEL_11;
    }
LABEL_24:
    sub_C8CC70((__int64)&v30, (__int64)&unk_4F8F810, (__int64)v18, v19, v16, v17);
  }
LABEL_11:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v35, (__int64)&v30);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v41, (__int64)&v36);
  if ( !v40 )
    _libc_free((unsigned __int64)v37);
  if ( !v34 )
    _libc_free((unsigned __int64)v31);
  return a1;
}
