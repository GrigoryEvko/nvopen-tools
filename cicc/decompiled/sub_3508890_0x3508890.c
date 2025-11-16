// Function: sub_3508890
// Address: 0x3508890
//
__int64 __fastcall sub_3508890(_QWORD *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 i; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 (__fastcall *v12)(__int64); // r12
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 result; // rax
  char v18; // al
  __int64 v19; // [rsp+8h] [rbp-138h]
  __int64 v21; // [rsp+18h] [rbp-128h]
  __int64 v22; // [rsp+20h] [rbp-120h]
  __int64 v23; // [rsp+28h] [rbp-118h]
  __int64 v24; // [rsp+30h] [rbp-110h]
  __int64 v25; // [rsp+38h] [rbp-108h]
  __int64 v26[8]; // [rsp+40h] [rbp-100h] BYREF
  _QWORD v27[12]; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+E0h] [rbp-60h]

  v2 = a2;
  v3 = *(_QWORD *)(a2 + 24);
  v4 = v3 + 48;
  for ( i = *(_QWORD *)(*(_QWORD *)(v3 + 56) + 32LL) + 40LL * (*(_DWORD *)(*(_QWORD *)(v3 + 56) + 40LL) & 0xFFFFFF);
        (*(_BYTE *)(v2 + 44) & 4) != 0;
        v2 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL )
  {
    ;
  }
  while ( 1 )
  {
    v6 = *(_QWORD *)(v2 + 32);
    v7 = v6 + 40LL * (*(_DWORD *)(v2 + 40) & 0xFFFFFF);
    if ( v6 != v7 )
      break;
    v2 = *(_QWORD *)(v2 + 8);
    if ( v4 == v2 )
      break;
    if ( (*(_BYTE *)(v2 + 44) & 4) == 0 )
    {
      v2 = v4;
      break;
    }
  }
  v26[3] = v7;
  v26[1] = v4;
  v26[4] = v4;
  v26[5] = v4;
  v26[6] = i;
  v26[7] = i;
  v26[0] = v2;
  v26[2] = v6;
  sub_3508760(v27, v26, (unsigned __int8 (__fastcall *)(__int64))sub_3507B50);
  v12 = (__int64 (__fastcall *)(__int64))v27[8];
  v13 = v27[0];
  v14 = v27[1];
  v25 = v27[4];
  v15 = v27[2];
  v16 = v27[3];
  v24 = v27[6];
  v22 = v27[7];
  v23 = v27[9];
  v21 = v27[11];
  result = v28;
  v19 = v28;
LABEL_8:
  if ( v13 != v23 )
  {
LABEL_9:
    if ( *(_BYTE *)v15
      || (v18 = *(_BYTE *)(v15 + 4), (v18 & 1) != 0)
      || (v18 & 2) != 0
      || (*(_BYTE *)(v15 + 3) & 0x10) != 0 && (*(_DWORD *)v15 & 0xFFF00) == 0 )
    {
      v9 = v15 + 40;
      result = v16;
      if ( v15 + 40 == v16 )
        goto LABEL_18;
    }
    else
    {
      sub_3507B80(a1, *(_DWORD *)(v15 + 8), v8, v9, v10, v11);
      v9 = v15 + 40;
      result = v16;
      if ( v15 + 40 == v16 )
      {
        while ( 1 )
        {
LABEL_18:
          v13 = *(_QWORD *)(v13 + 8);
          if ( v13 == v14 )
            goto LABEL_19;
          if ( (*(_BYTE *)(v13 + 44) & 4) == 0 )
            break;
          v16 = *(_QWORD *)(v13 + 32);
          result = v16 + 40LL * (*(_DWORD *)(v13 + 40) & 0xFFFFFF);
          if ( v16 != result )
            goto LABEL_19;
        }
        v13 = v14;
        goto LABEL_19;
      }
    }
    v16 = v9;
LABEL_19:
    v15 = v16;
    v16 = result;
    while ( 1 )
    {
LABEL_20:
      if ( v13 == v25 )
      {
LABEL_28:
        if ( v15 == v24 || v16 == v15 && v24 == v22 )
        {
          v13 = v25;
          if ( v25 == v23 )
            break;
          goto LABEL_9;
        }
      }
LABEL_21:
      result = v12(v15);
      if ( (_BYTE)result )
        goto LABEL_8;
      v15 += 40;
      if ( v16 == v15 )
      {
        while ( 1 )
        {
          v13 = *(_QWORD *)(v13 + 8);
          if ( v13 == v14 )
            break;
          if ( (*(_BYTE *)(v13 + 44) & 4) == 0 )
          {
            v13 = v14;
            if ( v14 != v25 )
              goto LABEL_21;
            goto LABEL_28;
          }
          v15 = *(_QWORD *)(v13 + 32);
          result = 5LL * (*(_DWORD *)(v13 + 40) & 0xFFFFFF);
          v16 = v15 + 40LL * (*(_DWORD *)(v13 + 40) & 0xFFFFFF);
          if ( v15 != v16 )
            goto LABEL_20;
        }
      }
    }
  }
  if ( v15 != v21 && (v16 != v15 || v21 != v19) )
    goto LABEL_9;
  return result;
}
