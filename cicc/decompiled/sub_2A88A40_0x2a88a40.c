// Function: sub_2A88A40
// Address: 0x2a88a40
//
_BYTE *__fastcall sub_2A88A40(_BYTE *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  unsigned __int64 v4; // r15
  __int64 v7; // rax
  _BYTE *v8; // r12
  __int64 **v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rdx
  int v12; // edx
  unsigned int v13; // eax
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 **v16; // rbx
  unsigned int v17; // esi
  _BYTE *result; // rax
  int v19; // ecx
  unsigned __int64 v20; // rax
  int v21; // edx
  char v22; // al
  int v23; // edx
  __int64 **v24; // rcx
  int v25; // ecx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rax
  int v31; // ecx
  int v32; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v33; // [rsp+0h] [rbp-A0h]
  int v34; // [rsp+18h] [rbp-88h]
  __int64 v35; // [rsp+20h] [rbp-80h] BYREF
  __int64 v36; // [rsp+28h] [rbp-78h]
  __int64 v37; // [rsp+30h] [rbp-70h] BYREF
  __int64 v38; // [rsp+38h] [rbp-68h]
  __int64 v39; // [rsp+40h] [rbp-60h] BYREF
  __int64 v40; // [rsp+48h] [rbp-58h]
  __int16 v41; // [rsp+60h] [rbp-40h]

  v4 = (unsigned __int64)a1;
  v7 = sub_B2BEC0(a4);
  v8 = (_BYTE *)v7;
  if ( *a1 <= 0x15u )
    v4 = sub_97B670(a1, v7, 0);
  v9 = *(__int64 ***)(v4 + 8);
  if ( *((_BYTE *)v9 + 8) == 18 && *(_BYTE *)(a2 + 8) == 17 )
  {
    v29 = (_QWORD *)a3[9];
    v37 = v4;
    v41 = 257;
    HIDWORD(v35) = 0;
    v30 = sub_BCB2E0(v29);
    v38 = sub_ACD640(v30, 0, 0);
    return (_BYTE *)sub_B35180((__int64)a3, a2, 0x17Du, (__int64)&v37, 2u, (unsigned int)v35, (__int64)&v39);
  }
  v35 = sub_9208B0((__int64)v8, *(_QWORD *)(v4 + 8));
  v36 = v10;
  v37 = sub_9208B0((__int64)v8, a2);
  v38 = v11;
  if ( v37 == v35 && (_BYTE)v38 == (_BYTE)v36 )
  {
    v21 = *((unsigned __int8 *)v9 + 8);
    v22 = *((_BYTE *)v9 + 8);
    if ( (unsigned int)(v21 - 17) > 1 )
    {
      if ( (_BYTE)v21 != 14 )
      {
        if ( v21 != 17 )
        {
LABEL_30:
          v23 = *(unsigned __int8 *)(a2 + 8);
          if ( (unsigned int)(v23 - 17) <= 1 )
            LOBYTE(v23) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
          v24 = (__int64 **)a2;
          if ( (_BYTE)v23 == 14 )
            v24 = (__int64 **)sub_AE4450((__int64)v8, a2);
          result = (_BYTE *)v4;
          if ( v9 != v24 )
          {
            v41 = 257;
            result = (_BYTE *)sub_2A882B0(a3, 0x31u, v4, v24, (__int64)&v39, 0, v34, 0);
          }
          v25 = *(unsigned __int8 *)(a2 + 8);
          if ( (unsigned int)(v25 - 17) <= 1 )
            LOBYTE(v25) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
          if ( (_BYTE)v25 == 14 )
          {
            v41 = 257;
            result = (_BYTE *)sub_2A882B0(a3, 0x30u, (unsigned __int64)result, (__int64 **)a2, (__int64)&v39, 0, v34, 0);
          }
LABEL_40:
          if ( *result != 5 )
            return result;
          return (_BYTE *)sub_97B670(result, (__int64)v8, 0);
        }
LABEL_50:
        v22 = *(_BYTE *)(*v9[2] + 8);
LABEL_28:
        if ( v22 == 14 )
        {
          v9 = (__int64 **)sub_AE4450((__int64)v8, (__int64)v9);
          v41 = 257;
          v4 = sub_2A882B0(a3, 0x2Fu, v4, v9, (__int64)&v39, 0, v34, 0);
        }
        goto LABEL_30;
      }
    }
    else if ( *(_BYTE *)(*v9[2] + 8) != 14 )
    {
      goto LABEL_27;
    }
    v31 = *(unsigned __int8 *)(a2 + 8);
    if ( (unsigned int)(v31 - 17) <= 1 )
      LOBYTE(v31) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
    if ( (_BYTE)v31 == 14 )
    {
      v41 = 257;
      result = (_BYTE *)sub_2A882B0(a3, 0x31u, v4, (__int64 **)a2, (__int64)&v39, 0, v34, 0);
      goto LABEL_40;
    }
LABEL_27:
    if ( (unsigned __int8)(v22 - 17) > 1u )
      goto LABEL_28;
    goto LABEL_50;
  }
  v12 = *((unsigned __int8 *)v9 + 8);
  if ( (unsigned int)(v12 - 17) > 1 )
  {
    if ( (_BYTE)v12 != 14 )
      goto LABEL_8;
  }
  else if ( *(_BYTE *)(*v9[2] + 8) != 14 )
  {
    goto LABEL_8;
  }
  v9 = (__int64 **)sub_AE4450((__int64)v8, (__int64)v9);
  v41 = 257;
  v20 = sub_2A882B0(a3, 0x2Fu, v4, v9, (__int64)&v39, 0, v34, 0);
  LOBYTE(v12) = *((_BYTE *)v9 + 8);
  v4 = v20;
LABEL_8:
  if ( (_BYTE)v12 != 12 )
  {
    v13 = sub_CA1930(&v35);
    v9 = (__int64 **)sub_BCCE00(*v9, v13);
    v41 = 257;
    v4 = sub_2A882B0(a3, 0x31u, v4, v9, (__int64)&v39, 0, v34, 0);
  }
  if ( *v8 )
  {
    v39 = sub_9208B0((__int64)v8, (__int64)v9);
    v40 = v26;
    v33 = (v39 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v39 = sub_9208B0((__int64)v8, a2);
    v40 = v27;
    v41 = 257;
    v28 = sub_AD64C0(*(_QWORD *)(v4 + 8), v33 - ((v39 + 7) & 0xFFFFFFFFFFFFFFF8LL), 0);
    v4 = sub_F94560(a3, v4, v28, (__int64)&v39, 0);
  }
  v14 = sub_CA1930(&v37);
  v15 = sub_BCCE00(*v9, v14);
  v41 = 257;
  v16 = (__int64 **)v15;
  v32 = sub_BCB060(*(_QWORD *)(v4 + 8));
  v17 = 49;
  if ( v32 != (unsigned int)sub_BCB060((__int64)v16) )
    v17 = 38;
  result = (_BYTE *)sub_2A882B0(a3, v17, v4, v16, (__int64)&v39, 0, v34, 0);
  if ( v16 != (__int64 **)a2 )
  {
    v19 = *(unsigned __int8 *)(a2 + 8);
    if ( (unsigned int)(v19 - 17) <= 1 )
      LOBYTE(v19) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
    v41 = 257;
    if ( (_BYTE)v19 == 14 )
      result = (_BYTE *)sub_2A882B0(a3, 0x30u, (unsigned __int64)result, (__int64 **)a2, (__int64)&v39, 0, v34, 0);
    else
      result = (_BYTE *)sub_2A882B0(a3, 0x31u, (unsigned __int64)result, (__int64 **)a2, (__int64)&v39, 0, v34, 0);
  }
  if ( *result <= 0x15u )
    return (_BYTE *)sub_97B670(result, (__int64)v8, 0);
  return result;
}
