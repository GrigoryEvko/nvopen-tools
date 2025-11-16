// Function: sub_27DE090
// Address: 0x27de090
//
__int64 __fastcall sub_27DE090(__int64 a1, char a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // rax
  bool v35; // zf
  _BYTE v36[8]; // [rsp+0h] [rbp-80h] BYREF
  unsigned __int64 v37; // [rsp+8h] [rbp-78h]
  char v38; // [rsp+1Ch] [rbp-64h]
  unsigned __int64 v39; // [rsp+38h] [rbp-48h]
  char v40; // [rsp+4Ch] [rbp-34h]

  result = sub_27DD130((__int64 *)a1);
  if ( !result )
  {
    if ( a2 )
    {
      if ( *(_BYTE *)(a1 + 88) )
      {
        *(_BYTE *)(a1 + 88) = 0;
        sub_27DD0B0((__int64)v36, a1, v4, v5, v6, v7);
        sub_27DC6B0((__int64)v36, (__int64)&unk_4F8E5A8, v9, v10, v11, v12);
        sub_27DC6B0((__int64)v36, (__int64)&unk_4F8D9A8, v13, v14, v15, v16);
        v17 = *(_QWORD *)a1;
        sub_BBE020(*(_QWORD *)(a1 + 8), *(_QWORD *)a1, (__int64)v36, v18);
        v19 = *(_QWORD *)(a1 + 48);
        sub_FFCE90(v19, v17, v20, v21, v22, v23);
        sub_FFD870(v19, v17, v24, v25, v26, v27);
        sub_FFBC40(v19, v17);
        v8 = sub_BC1CD0(*(_QWORD *)(a1 + 8), &unk_4F8E5A8, *(_QWORD *)a1) + 8;
        v28 = sub_BC1CD0(*(_QWORD *)(a1 + 8), &unk_4F89C30, *(_QWORD *)a1);
        v29 = *(_QWORD *)(a1 + 8);
        v30 = *(_QWORD *)a1;
        *(_QWORD *)(a1 + 24) = v28 + 8;
        v31 = sub_BC1CD0(v29, &unk_4F6D3F8, v30);
        v32 = *(_QWORD *)(a1 + 8);
        v33 = *(_QWORD *)a1;
        *(_QWORD *)(a1 + 16) = v31 + 8;
        v34 = sub_BC1CD0(v32, &unk_4F86540, v33);
        v35 = v40 == 0;
        *(_QWORD *)(a1 + 40) = v34 + 8;
        if ( v35 )
          _libc_free(v39);
        if ( !v38 )
          _libc_free(v37);
      }
      else
      {
        v8 = sub_BC1CD0(*(_QWORD *)(a1 + 8), &unk_4F8E5A8, *(_QWORD *)a1) + 8;
      }
      *(_QWORD *)(a1 + 72) = v8;
      *(_BYTE *)(a1 + 80) = 1;
      return v8;
    }
    else
    {
      return *(_QWORD *)(a1 + 72);
    }
  }
  return result;
}
