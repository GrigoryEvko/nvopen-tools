// Function: sub_325B6A0
// Address: 0x325b6a0
//
__int64 __fastcall sub_325B6A0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  char v4; // r12
  __int64 v5; // rbx
  char v6; // al
  __int64 v7; // rax
  __int64 v8; // rcx
  bool v9; // al
  __int64 v10; // rsi
  __int64 result; // rax
  __int64 v12; // r13
  const char *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rcx
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // r9
  unsigned int v20; // eax
  unsigned __int8 *v21; // r9
  __int64 v22; // r10
  int v23; // eax
  int v24; // eax
  unsigned __int8 *v25; // r9
  char v26; // al
  int v27; // [rsp+4h] [rbp-4Ch]
  __int64 v28; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v29; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  __int64 v32; // [rsp+18h] [rbp-38h]

  *(_WORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 26) = 0;
  v3 = *(_QWORD *)(a2 + 440);
  v4 = *(_BYTE *)(a2 + 580);
  v5 = *(_QWORD *)a2;
  v32 = *(_QWORD *)(a2 + 432);
  v6 = sub_31DB790(*(_QWORD *)(a1 + 8));
  if ( v6 )
    v6 = *(_BYTE *)(a2 + 343);
  *(_BYTE *)(a1 + 26) = v6;
  v7 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  v8 = 0;
  if ( (*(_BYTE *)(v5 + 2) & 8) == 0 )
    goto LABEL_4;
  v28 = v7;
  v27 = *(_DWORD *)(v7 + 940);
  v18 = (unsigned __int8 *)sub_B2E500(v5);
  v19 = sub_BD3990(v18, a2);
  if ( *v19 )
    v19 = 0;
  v30 = v19;
  v20 = sub_B2A630((__int64)v19);
  v21 = v30;
  v22 = v28;
  v8 = v20;
  if ( (*(_BYTE *)(v5 + 2) & 8) != 0 && !v20 )
  {
    v24 = sub_A746B0((_QWORD *)(v5 + 120));
    v25 = v30;
    v22 = v28;
    if ( v24
      || (v31 = v28, v29 = v25, v26 = sub_B2D610(v5, 41), v22 = v31, !v26)
      || (v21 = v29, v8 = 0, (*(_BYTE *)(v5 + 2) & 8) != 0) )
    {
      *(_BYTE *)(a1 + 24) = 1;
      v8 = 0;
      v23 = *(_DWORD *)(v22 + 944);
      goto LABEL_25;
    }
  }
  if ( (v32 != v3 || v4) && v27 != 255 && v21 )
  {
    *(_BYTE *)(a1 + 24) = 1;
    v23 = *(_DWORD *)(v22 + 944);
LABEL_25:
    v9 = v23 != 255;
    goto LABEL_5;
  }
LABEL_4:
  *(_BYTE *)(a1 + 24) = 0;
  v9 = 0;
LABEL_5:
  v10 = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(a1 + 25) = v9;
  result = *(_QWORD *)(v10 + 208);
  if ( *(_DWORD *)(result + 336) == 4 )
  {
    result = *(unsigned int *)(result + 344);
    if ( (_DWORD)result )
    {
      if ( (_DWORD)result != 6 )
        return (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __int64))(*(_QWORD *)a1 + 104LL))(
                 a1,
                 *(_QWORD *)(a2 + 328),
                 *(_QWORD *)(v10 + 280),
                 v8);
    }
  }
  if ( (_DWORD)v8 == 7 && v4 != 1 )
  {
    v12 = *(_QWORD *)(a2 + 88);
    v13 = sub_BD5D20(*(_QWORD *)a2);
    v16 = (__int64)v13;
    v17 = v14;
    if ( v14 )
    {
      if ( *v13 == 1 )
      {
        v17 = v14 - 1;
        v16 = (__int64)(v13 + 1);
      }
    }
    result = sub_325B570(a1, v12, v16, v17, v15, v16);
  }
  *(_BYTE *)(a1 + 25) = v4;
  *(_BYTE *)(a1 + 24) = 0;
  return result;
}
