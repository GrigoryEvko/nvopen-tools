// Function: sub_A84D20
// Address: 0xa84d20
//
__int64 __fastcall sub_A84D20(_QWORD *a1)
{
  __int64 v2; // r12
  unsigned int v3; // r15d
  int v4; // ebx
  __int64 v5; // rax
  unsigned __int8 v6; // dl
  __int64 v7; // rax
  _BYTE *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int8 v12; // dl
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  _QWORD *v16; // rax
  int v17; // ebx
  __int64 v18; // rax
  __int64 result; // rax
  __int64 v20; // rdi
  __int64 v21; // rdi
  unsigned __int8 v22; // [rsp-69h] [rbp-69h]
  unsigned __int8 v23; // [rsp-59h] [rbp-59h] BYREF
  void *v24; // [rsp-58h] [rbp-58h] BYREF
  __int64 v25; // [rsp-50h] [rbp-50h]
  _QWORD *v26; // [rsp-48h] [rbp-48h]
  int v27; // [rsp-40h] [rbp-40h]

  if ( byte_4F80CE8 )
    return 0;
  v2 = a1[108];
  if ( !v2 )
    goto LABEL_25;
  v3 = 0;
  v4 = sub_B91A00(a1[108]);
  if ( v4 )
  {
    do
    {
      v5 = sub_B91A10(v2, v3);
      v6 = *(_BYTE *)(v5 - 16);
      if ( (v6 & 2) != 0 )
      {
        if ( *(_DWORD *)(v5 - 24) <= 2u )
          continue;
        v7 = *(_QWORD *)(v5 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(v5 - 16) >> 6) & 0xFu) <= 2 )
          continue;
        v7 = -16 - 8LL * ((v6 >> 2) & 0xF) + v5;
      }
      v8 = *(_BYTE **)(v7 + 8);
      if ( v8 )
      {
        if ( !*v8 )
        {
          v9 = sub_B91420(v8, v3);
          if ( v10 == 18
            && !(*(_QWORD *)v9 ^ 0x6E49206775626544LL | *(_QWORD *)(v9 + 8) ^ 0x6973726556206F66LL)
            && *(_WORD *)(v9 + 16) == 28271 )
          {
            break;
          }
        }
      }
    }
    while ( v4 != ++v3 );
  }
  if ( (unsigned int)sub_B91A00(v2) != v3
    && ((v11 = sub_B91A10(v2, v3), v12 = *(_BYTE *)(v11 - 16), (v12 & 2) != 0)
      ? (v13 = *(_QWORD *)(v11 - 32))
      : (v13 = v11 - 8LL * ((v12 >> 2) & 0xF) - 16),
        (v14 = *(_QWORD *)(v13 + 16)) != 0 && *(_BYTE *)v14 == 1 && (v15 = *(_QWORD *)(v14 + 136), *(_BYTE *)v15 == 17)) )
  {
    v16 = *(_QWORD **)(v15 + 24);
    if ( *(_DWORD *)(v15 + 32) > 0x40u )
      v16 = (_QWORD *)*v16;
    v17 = (int)v16;
    if ( (_DWORD)v16 == 3 )
    {
      v23 = 0;
      v18 = sub_CB72A0();
      if ( (unsigned __int8)sub_C09360(a1, v18, &v23) )
        sub_C64ED0("Broken module found, compilation aborted!", 1);
      result = v23;
      if ( v23 )
      {
        v20 = *a1;
        v26 = a1;
        v25 = 0x100000008LL;
        v24 = &unk_49D9C48;
        sub_B6EB20(v20, &v24);
        return sub_AEB840(a1);
      }
      return result;
    }
  }
  else
  {
LABEL_25:
    v17 = 0;
  }
  result = sub_AEB840(a1);
  if ( (_BYTE)result )
  {
    v21 = *a1;
    v22 = result;
    v26 = a1;
    v25 = 0x100000008LL;
    v27 = v17;
    v24 = &unk_49D9C18;
    sub_B6EB20(v21, &v24);
    return v22;
  }
  return result;
}
