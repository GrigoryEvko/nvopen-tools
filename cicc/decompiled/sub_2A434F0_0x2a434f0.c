// Function: sub_2A434F0
// Address: 0x2a434f0
//
__int64 __fastcall sub_2A434F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r14
  __int64 v8; // rax
  int v10; // edi
  void **v11; // rsi
  __int64 *v12; // rdx
  __int64 **v13; // rax
  void *v14; // rdx
  void **v15; // rax
  __int64 v16; // [rsp+10h] [rbp-90h] BYREF
  void **v17; // [rsp+18h] [rbp-88h]
  int v18; // [rsp+20h] [rbp-80h]
  int v19; // [rsp+24h] [rbp-7Ch]
  int v20; // [rsp+28h] [rbp-78h]
  char v21; // [rsp+2Ch] [rbp-74h]
  void *v22; // [rsp+30h] [rbp-70h] BYREF
  void *v23; // [rsp+38h] [rbp-68h] BYREF
  __int64 v24; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v25; // [rsp+48h] [rbp-58h]
  __int64 v26; // [rsp+50h] [rbp-50h]
  int v27; // [rsp+58h] [rbp-48h]
  char v28; // [rsp+5Ch] [rbp-44h]
  _BYTE v29[64]; // [rsp+60h] [rbp-40h] BYREF

  v7 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v8 = sub_BC1CD0(a4, &unk_4F8F810, a3);
  if ( (unsigned __int8)sub_2A41FF0(a3, v7 + 8, *(_QWORD *)(v8 + 8)) )
  {
    v18 = 2;
    v17 = &v22;
    v20 = 0;
    v21 = 1;
    v24 = 0;
    v25 = v29;
    v26 = 2;
    v27 = 0;
    v28 = 1;
    v19 = 1;
    v22 = &unk_4F81450;
    v16 = 1;
    if ( &unk_4F81450 == (_UNKNOWN *)&qword_4F82400 || &unk_4F81450 == &unk_4F8F810 )
    {
      v10 = 1;
      v11 = &v23;
    }
    else
    {
      v19 = 2;
      v10 = 2;
      v16 = 2;
      v23 = &unk_4F8F810;
      v11 = (void **)&v24;
    }
    v12 = (__int64 *)&unk_4F81450;
    v13 = (__int64 **)&v22;
    while ( v12 != &qword_4F82400 )
    {
      if ( ++v13 == (__int64 **)v11 )
      {
        v14 = &unk_4F81450;
        v15 = &v22;
        while ( v14 != &unk_4F82408 )
        {
          if ( ++v15 == v11 )
          {
            if ( v10 == 1 )
            {
              v19 = 2;
              *v15 = &unk_4F82408;
              ++v16;
            }
            else
            {
              sub_C8CC70(
                (__int64)&v16,
                (__int64)&unk_4F82408,
                (__int64)v14,
                (__int64)&v22,
                (__int64)&v16,
                (__int64)&unk_4F82408);
            }
            goto LABEL_11;
          }
          v14 = *v15;
        }
        break;
      }
      v12 = *v13;
    }
LABEL_11:
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)&v22, (__int64)&v16);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v29, (__int64)&v24);
    if ( !v28 )
      _libc_free((unsigned __int64)v25);
    if ( !v21 )
      _libc_free((unsigned __int64)v17);
  }
  else
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
  }
  return a1;
}
