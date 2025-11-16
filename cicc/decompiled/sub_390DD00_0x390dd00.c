// Function: sub_390DD00
// Address: 0x390dd00
//
__int64 __fastcall sub_390DD00(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 *v3; // rax
  __int64 (__fastcall *v4)(__int64, unsigned __int8 *); // rcx
  char v5; // cl
  __int64 result; // rax
  unsigned int v7; // r15d
  _QWORD **v8; // rax
  __int64 v9; // rcx
  _QWORD **v10; // r12
  _QWORD *v11; // rbx
  _QWORD **v12; // r13
  __int64 v13; // rcx
  __int64 (*v14)(); // rax
  _QWORD **v15; // rax
  char v16; // al
  char v17; // al
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+8h] [rbp-38h]
  _BYTE *v20; // [rsp+8h] [rbp-38h]
  _BYTE *v21; // [rsp+8h] [rbp-38h]
  _BYTE *v22; // [rsp+8h] [rbp-38h]

  v3 = *(__int64 **)a1;
  *(_QWORD *)(a1 + 160) = a2;
  v4 = (__int64 (__fastcall *)(__int64, unsigned __int8 *))v3[2];
  if ( v4 == sub_390D9D0 )
  {
    v5 = *a3;
  }
  else
  {
    v22 = a3;
    v17 = v4(a1, a3);
    a3 = v22;
    v5 = v17;
    v3 = *(__int64 **)a1;
  }
  *(_BYTE *)(a1 + 8) = v5;
  result = *v3;
  if ( (__int64 (*)())result == sub_390D9B0 )
  {
    if ( !v5 )
      return result;
    v7 = 0;
    goto LABEL_7;
  }
  v21 = a3;
  result = ((__int64 (__fastcall *)(__int64, _BYTE *))result)(a1, a3);
  v13 = 0;
  a3 = v21;
  v7 = result;
  if ( *(_BYTE *)(a1 + 8) )
  {
LABEL_7:
    v8 = *(_QWORD ***)(a1 + 32);
    if ( v8 == *(_QWORD ***)(a1 + 24) )
      v9 = *(unsigned int *)(a1 + 44);
    else
      v9 = *(unsigned int *)(a1 + 40);
    v10 = &v8[v9];
    if ( v8 == v10 )
      goto LABEL_12;
    while ( 1 )
    {
      v11 = *v8;
      v12 = v8;
      if ( (unsigned __int64)*v8 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++v8 )
        goto LABEL_12;
    }
    if ( v8 == v10 )
    {
LABEL_12:
      result = v7;
      v13 = 0;
    }
    else
    {
      v13 = 0;
      v14 = *(__int64 (**)())(*v11 + 24LL);
      if ( v14 != sub_390D9E0 )
        goto LABEL_25;
      while ( 1 )
      {
        v15 = v12 + 1;
        if ( v12 + 1 == v10 )
          break;
        while ( 1 )
        {
          v11 = *v15;
          v12 = v15;
          if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v10 == ++v15 )
            goto LABEL_18;
        }
        if ( v15 == v10 )
          break;
        v14 = *(__int64 (**)())(*v11 + 24LL);
        if ( v14 != sub_390D9E0 )
        {
LABEL_25:
          v18 = v13;
          v20 = a3;
          v16 = ((__int64 (__fastcall *)(_QWORD *, _BYTE *))v14)(v11, a3);
          a3 = v20;
          v13 = v18;
          if ( v16 )
            v13 = v11[1] | v18;
        }
      }
LABEL_18:
      LOBYTE(v15) = v13 != 0;
      result = v7 | (unsigned int)v15;
    }
  }
  v19 = v13;
  if ( (_BYTE)result )
  {
    result = sub_38D57A0(a2);
    if ( (_BYTE)v7 )
      *(_BYTE *)(result + 56) = 1;
    *(_QWORD *)(result + 48) |= v19;
  }
  return result;
}
