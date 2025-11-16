// Function: sub_F9A0D0
// Address: 0xf9a0d0
//
__int64 __fastcall sub_F9A0D0(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r13
  _QWORD *v9; // rdi
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  const char *v14; // rsi
  unsigned __int64 v15; // rax
  int v16; // eax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v23; // [rsp+8h] [rbp-A8h]
  __int64 v24; // [rsp+10h] [rbp-A0h]
  __int64 v25; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v26; // [rsp+10h] [rbp-A0h]
  __int64 v28; // [rsp+28h] [rbp-88h] BYREF
  unsigned __int64 v29; // [rsp+30h] [rbp-80h] BYREF
  int v30; // [rsp+38h] [rbp-78h]
  unsigned __int64 v31; // [rsp+40h] [rbp-70h]
  int v32; // [rsp+48h] [rbp-68h]
  const char *v33; // [rsp+50h] [rbp-60h] BYREF
  __int64 v34; // [rsp+58h] [rbp-58h]
  _QWORD v35[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v36; // [rsp+70h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 40);
  v5 = *(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL);
  v28 = v5;
  if ( a3 )
    sub_AA5980(v5, v4, 0);
  v23 = *(_QWORD *)(v4 + 72);
  v33 = sub_BD5D20(v4);
  v36 = 773;
  v34 = v6;
  v35[0] = ".unreachabledefault";
  v24 = sub_AA48A0(v4);
  v7 = sub_22077B0(80);
  v8 = v7;
  if ( v7 )
    sub_AA4D50(v7, v24, (__int64)&v33, v23, v5);
  v25 = sub_BD5C60(a1);
  sub_B43C20((__int64)&v33, v8);
  v9 = sub_BD2C40(72, unk_3F148B8);
  if ( v9 )
    sub_B4C8A0((__int64)v9, v25, (__int64)v33, v34);
  result = *(_QWORD *)(a1 - 8);
  if ( *(_QWORD *)(result + 32) )
  {
    v11 = *(_QWORD *)(result + 40);
    **(_QWORD **)(result + 48) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(result + 48);
  }
  *(_QWORD *)(result + 32) = v8;
  if ( v8 )
  {
    v12 = *(_QWORD *)(v8 + 16);
    *(_QWORD *)(result + 40) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = result + 40;
    *(_QWORD *)(result + 48) = v8 + 16;
    result += 32;
    *(_QWORD *)(v8 + 16) = result;
  }
  if ( a2 )
  {
    v35[0] = v4;
    v33 = (const char *)v35;
    v35[1] = v8 & 0xFFFFFFFFFFFFFFFBLL;
    v34 = 0x200000001LL;
    if ( !a3 )
      goto LABEL_16;
    v15 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v15 == v4 + 48 )
      goto LABEL_26;
    if ( !v15 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
    {
LABEL_26:
      v16 = 0;
      v18 = 0;
      v17 = 0;
    }
    else
    {
      v26 = v15 - 24;
      v16 = sub_B46E30(v15 - 24);
      v17 = v26;
      v18 = v26;
    }
    v29 = v17;
    v30 = 0;
    v31 = v18;
    v32 = v16;
    if ( sub_F99F90((__int64)&v29, &v28) )
    {
LABEL_16:
      v13 = 1;
      v14 = (const char *)v35;
    }
    else
    {
      sub_F35FA0((__int64)&v33, v4, v5 | 4, v19, v20, v21);
      v14 = v33;
      v13 = (unsigned int)v34;
    }
    result = sub_FFB3D0(a2, v14, v13);
    if ( v33 != (const char *)v35 )
      return _libc_free(v33, v14);
  }
  return result;
}
