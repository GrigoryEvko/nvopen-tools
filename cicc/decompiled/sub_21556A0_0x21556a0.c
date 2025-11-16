// Function: sub_21556A0
// Address: 0x21556a0
//
__int64 __fastcall sub_21556A0(__int64 a1, unsigned int *a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  _BYTE *v9; // rax
  __int64 v10; // rsi
  _BYTE *v11; // r8
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rcx
  __int64 v16; // rdx
  _BYTE *v17; // [rsp+8h] [rbp-78h]
  _QWORD v18[2]; // [rsp+10h] [rbp-70h] BYREF
  char v19; // [rsp+20h] [rbp-60h]
  unsigned int *v20[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v21; // [rsp+40h] [rbp-40h]

  v3 = a1 + 80;
  v4 = a1 + 8;
  v5 = a1 + 64;
  *(_DWORD *)(v5 - 56) = 0;
  *(_QWORD *)(v5 - 48) = 0;
  *(_QWORD *)(v5 - 40) = v4;
  *(_QWORD *)(v5 - 32) = v4;
  *(_QWORD *)(v5 - 24) = 0;
  *(_QWORD *)(v5 - 8) = 0;
  *(_QWORD *)(a1 + 64) = v3;
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  sub_2240AE0(v5, a2);
  v20[0] = a2;
  v21 = 260;
  result = sub_16C2DE0((__int64)v18, (__int64)v20, 0xFFFFFFFFFFFFFFFFLL, 1, 0);
  if ( (v19 & 1) == 0 )
  {
    v8 = v18[0];
    v7 = *(_QWORD *)(a1 + 56);
    v18[0] = 0;
    *(_QWORD *)(a1 + 56) = v8;
    if ( !v7 )
      goto LABEL_9;
    goto LABEL_7;
  }
  v7 = *(_QWORD *)(a1 + 56);
  if ( LODWORD(v18[0]) )
  {
    *(_QWORD *)(a1 + 56) = 0;
    if ( v7 )
    {
      result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
      *(_BYTE *)(a1 + 48) = 1;
      if ( (v19 & 1) == 0 )
      {
        if ( v18[0] )
          return (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v18[0] + 8LL))(v18[0]);
      }
    }
    else
    {
      *(_BYTE *)(a1 + 48) = 1;
    }
    return result;
  }
  v8 = v18[0];
  v18[0] = 0;
  *(_QWORD *)(a1 + 56) = v8;
  if ( v7 )
  {
LABEL_7:
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
    if ( (v19 & 1) == 0 && v18[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v18[0] + 8LL))(v18[0]);
    v8 = *(_QWORD *)(a1 + 56);
  }
LABEL_9:
  *(_BYTE *)(a1 + 48) = *(_QWORD *)(v8 + 16) == *(_QWORD *)(v8 + 8);
  v9 = sub_21546A0(a1, *(_BYTE **)(v8 + 8));
  LODWORD(v18[0]) = 1;
  v10 = v4;
  v11 = v9;
  v12 = *(_QWORD *)(a1 + 16);
  v14 = v13;
  if ( !v12 )
    goto LABEL_16;
  do
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(v12 + 16);
      v16 = *(_QWORD *)(v12 + 24);
      if ( *(_DWORD *)(v12 + 32) <= 1u )
        break;
      v12 = *(_QWORD *)(v12 + 24);
      if ( !v16 )
        goto LABEL_14;
    }
    v10 = v12;
    v12 = *(_QWORD *)(v12 + 16);
  }
  while ( v15 );
LABEL_14:
  if ( v4 == v10 || (result = *(unsigned int *)(v10 + 32), !(_DWORD)result) )
  {
LABEL_16:
    v17 = v11;
    v20[0] = (unsigned int *)v18;
    result = sub_21555E0((_QWORD *)a1, v10, v20);
    v11 = v17;
    v10 = result;
  }
  *(_QWORD *)(v10 + 40) = v11;
  *(_QWORD *)(v10 + 48) = v14;
  *(_BYTE *)(v10 + 56) = 0;
  return result;
}
