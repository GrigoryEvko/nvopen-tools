// Function: sub_3060450
// Address: 0x3060450
//
_QWORD *__fastcall sub_3060450(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rax
  _QWORD *v5; // r12
  __int64 *v6; // r14
  unsigned int v7; // r9d
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rcx
  size_t v13; // r13
  _BYTE *v14; // rdi
  _BYTE *v15; // r10
  __int64 v16; // rax
  _BYTE *v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  unsigned int v20; // [rsp+1Ch] [rbp-44h]
  unsigned int v21; // [rsp+1Ch] [rbp-44h]
  unsigned int v22; // [rsp+1Ch] [rbp-44h]
  size_t v23[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = a3[16];
  a3[26] += 280LL;
  v5 = (_QWORD *)((v4 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( a3[17] >= (unsigned __int64)(v5 + 35) && v4 )
    a3[16] = v5 + 35;
  else
    v5 = (_QWORD *)sub_9D1E70((__int64)(a3 + 16), 280, 280, 3);
  v6 = v5 + 3;
  v5[1] = v5 + 3;
  *v5 = &unk_4A30878;
  v5[2] = 0x800000000LL;
  v7 = *(_DWORD *)(a1 + 16);
  if ( v7 && v5 + 1 != (_QWORD *)(a1 + 8) )
  {
    v9 = v7;
    if ( v7 > 8 )
    {
      v20 = *(_DWORD *)(a1 + 16);
      sub_95D880((__int64)(v5 + 1), v7);
      v6 = (__int64 *)v5[1];
      v9 = *(unsigned int *)(a1 + 16);
      v7 = v20;
    }
    v10 = *(_QWORD *)(a1 + 8);
    v11 = 32 * v9;
    v12 = v10 + v11;
    if ( v10 == v10 + v11 )
    {
LABEL_10:
      *((_DWORD *)v5 + 4) = v7;
      goto LABEL_5;
    }
    while ( 1 )
    {
      if ( !v6 )
        goto LABEL_13;
      *v6 = (__int64)(v6 + 2);
      v15 = *(_BYTE **)v10;
      v13 = *(_QWORD *)(v10 + 8);
      if ( v13 + *(_QWORD *)v10 && !v15 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v23[0] = *(_QWORD *)(v10 + 8);
      if ( v13 > 0xF )
        break;
      v14 = (_BYTE *)*v6;
      if ( v13 == 1 )
      {
        *v14 = *v15;
        v13 = v23[0];
        v14 = (_BYTE *)*v6;
      }
      else if ( v13 )
      {
        goto LABEL_22;
      }
LABEL_12:
      v6[1] = v13;
      v14[v13] = 0;
LABEL_13:
      v10 += 32;
      v6 += 4;
      if ( v12 == v10 )
        goto LABEL_10;
    }
    v17 = v15;
    v18 = v12;
    v21 = v7;
    v16 = sub_22409D0((__int64)v6, v23, 0);
    v7 = v21;
    v12 = v18;
    *v6 = v16;
    v14 = (_BYTE *)v16;
    v15 = v17;
    v6[2] = v23[0];
LABEL_22:
    v19 = v12;
    v22 = v7;
    memcpy(v14, v15, v13);
    v13 = v23[0];
    v14 = (_BYTE *)*v6;
    v12 = v19;
    v7 = v22;
    goto LABEL_12;
  }
LABEL_5:
  a3[5] = v5;
  return v5;
}
