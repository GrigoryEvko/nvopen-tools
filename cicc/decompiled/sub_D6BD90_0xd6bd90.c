// Function: sub_D6BD90
// Address: 0xd6bd90
//
__int64 __fastcall sub_D6BD90(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rcx
  bool v16; // zf
  __int64 *v17; // rax
  _QWORD *v18; // rax
  __int64 result; // rax
  unsigned int v20; // edx
  unsigned int v21; // esi
  unsigned int v22; // edi
  __int64 v23; // rdx
  __int64 v24; // [rsp-10h] [rbp-D0h]
  __int64 v25; // [rsp+10h] [rbp-B0h]
  __int64 v26; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v27; // [rsp+28h] [rbp-98h] BYREF
  __int64 *v28; // [rsp+30h] [rbp-90h] BYREF
  __int64 *v29; // [rsp+38h] [rbp-88h] BYREF
  __int64 v30; // [rsp+40h] [rbp-80h] BYREF
  __int64 v31; // [rsp+48h] [rbp-78h]
  __int64 v32; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v33; // [rsp+58h] [rbp-68h]
  char v34; // [rsp+90h] [rbp-30h] BYREF

  v4 = &v32;
  v26 = a2;
  v30 = 0;
  v31 = 1;
  do
  {
    *v4 = -4096;
    v4 += 2;
  }
  while ( v4 != (__int64 *)&v34 );
  v8 = v26;
  v9 = sub_D68B40(*a1, v26);
  v27 = v9;
  v10 = v9;
  if ( !v9 )
    goto LABEL_12;
  v11 = *(_QWORD *)(v9 - 8);
  v12 = 0x1FFFFFFFE0LL;
  v13 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
  if ( v13 )
  {
    v14 = 0;
    v15 = v11 + 32LL * *(unsigned int *)(v10 + 76);
    do
    {
      if ( a3 == *(_QWORD *)(v15 + 8 * v14) )
      {
        v12 = 32 * v14;
        goto LABEL_9;
      }
      ++v14;
    }
    while ( v13 != (_DWORD)v14 );
    v12 = 0x1FFFFFFFE0LL;
  }
LABEL_9:
  v25 = *(_QWORD *)(v11 + v12);
  v16 = (unsigned __int8)sub_D69D00((__int64)&v30, &v27, &v28) == 0;
  v17 = v28;
  if ( v16 )
  {
    ++v30;
    v29 = v28;
    v20 = ((unsigned int)v31 >> 1) + 1;
    if ( (v31 & 1) != 0 )
    {
      v22 = 12;
      v21 = 4;
    }
    else
    {
      v21 = v33;
      v22 = 3 * v33;
    }
    if ( v22 <= 4 * v20 )
    {
      sub_D6B9C0((__int64)&v30, 2 * v21);
    }
    else
    {
      if ( v21 - (v20 + HIDWORD(v31)) > v21 >> 3 )
      {
LABEL_19:
        LODWORD(v31) = v31 & 1 | (2 * v20);
        if ( *v17 != -4096 )
          --HIDWORD(v31);
        v23 = v27;
        v17[1] = 0;
        v18 = v17 + 1;
        *(v18 - 1) = v23;
        goto LABEL_11;
      }
      sub_D6B9C0((__int64)&v30, v21);
    }
    sub_D69D00((__int64)&v30, &v27, &v29);
    v17 = v29;
    v20 = ((unsigned int)v31 >> 1) + 1;
    goto LABEL_19;
  }
  v18 = v28 + 1;
LABEL_11:
  *v18 = v25;
  v8 = v26;
LABEL_12:
  v29 = &v26;
  sub_D69A00(
    a1,
    v8,
    a3,
    a4,
    (__int64)&v30,
    1,
    (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_D677B0,
    (__int64)&v29);
  result = v24;
  if ( (v31 & 1) == 0 )
    return sub_C7D6A0(v32, 16LL * v33, 8);
  return result;
}
