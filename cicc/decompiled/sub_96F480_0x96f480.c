// Function: sub_96F480
// Address: 0x96f480
//
__int64 __fastcall sub_96F480(unsigned int a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbp
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 result; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rcx
  int v16; // eax
  unsigned __int8 *v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned __int8 *v20; // rdi
  __int16 v21; // ax
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // [rsp-68h] [rbp-68h]
  unsigned int v33; // [rsp-50h] [rbp-50h]
  __int64 v34; // [rsp-50h] [rbp-50h]
  __int64 v35; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v36; // [rsp-40h] [rbp-40h]
  __int64 v37; // [rsp-28h] [rbp-28h]
  __int64 v38; // [rsp-20h] [rbp-20h]
  __int64 v39; // [rsp-18h] [rbp-18h]
  __int64 v40; // [rsp-10h] [rbp-10h]
  __int64 v41; // [rsp-8h] [rbp-8h]

  v41 = v4;
  v40 = v8;
  v39 = v7;
  v38 = v6;
  v37 = v5;
  switch ( a1 )
  {
    case '&':
    case '\'':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case '-':
    case '.':
    case '2':
      goto LABEL_3;
    case '/':
      if ( *(_BYTE *)a2 != 5 )
        goto LABEL_3;
      v21 = *(_WORD *)(a2 + 2);
      if ( v21 == 48 )
      {
        v27 = sub_AE4450(a4, *(_QWORD *)(a2 + 8));
        v23 = sub_96F3F0(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v27, 0, a4);
        goto LABEL_28;
      }
      if ( v21 != 34 )
        goto LABEL_3;
      v36 = sub_AE43F0(a4, *(_QWORD *)(a2 + 8));
      if ( v36 > 0x40 )
        sub_C43690(&v35, 0, 0);
      else
        v35 = 0;
      v22 = sub_BD45C0(a2, a4, (unsigned int)&v35, 1, 0, 0, 0, 0);
      if ( (unsigned __int8)sub_AC30F0(v22) )
      {
        v28 = sub_BD5C60(a2, a4, v32);
        v23 = sub_ACCFD0(v28, &v35);
      }
      else
      {
        if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 2 )
          goto LABEL_24;
        v24 = sub_BB5290(a2, a4, v32);
        if ( !(unsigned __int8)sub_BCAC40(v24, 8) )
          goto LABEL_24;
        v25 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL);
        v26 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
        if ( *(_BYTE *)v26 != 5 )
        {
          sub_AE4570(a4, v25);
LABEL_24:
          v23 = 0;
          goto LABEL_25;
        }
        v34 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
        v29 = sub_AE4570(a4, v25);
        if ( *(_QWORD *)(v26 + 8) != v29
          || *(_WORD *)(v26 + 2) != 15
          || !(unsigned __int8)sub_AC30F0(*(_QWORD *)(v26 - 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF))) )
        {
          goto LABEL_24;
        }
        v30 = *(_QWORD *)(v26 + 32 * (1LL - (*(_DWORD *)(v26 + 4) & 0x7FFFFFF)));
        v31 = sub_AD4C50(v34, v29, 0);
        v23 = sub_AD57F0(v31, v30, 0, 0);
      }
LABEL_25:
      if ( v36 > 0x40 && v35 )
        j_j___libc_free_0_0(v35);
LABEL_28:
      if ( v23 )
        return sub_96F3F0(v23, a3, 0, a4);
LABEL_3:
      if ( (unsigned __int8)sub_AC4810(a1) )
        return sub_ADAB70(a1, a2, a3, 0);
      else
        return sub_AA93C0(a1, a2, a3);
    case '0':
      if ( *(_BYTE *)a2 != 5 )
        goto LABEL_3;
      if ( *(_WORD *)(a2 + 2) != 47 )
        goto LABEL_3;
      v13 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v33 = sub_AE43A0(a4, *(_QWORD *)(v13 + 8));
      if ( v33 > (unsigned int)sub_BCB060(*(_QWORD *)(a2 + 8)) )
        goto LABEL_3;
      v14 = *(_QWORD *)(v13 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
        v14 = **(_QWORD **)(v14 + 16);
      v15 = a3;
      v16 = *(_DWORD *)(v14 + 8) >> 8;
      if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
        v15 = **(_QWORD **)(a3 + 16);
      if ( *(_DWORD *)(v15 + 8) >> 8 != v16 )
        goto LABEL_3;
      v17 = *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      result = sub_96E500(v17, a3, a4);
      if ( result )
        return result;
      v18 = a4;
      v19 = a3;
      v20 = v17;
      return sub_96F860(v20, v19, v18);
    case '1':
      result = sub_96E500((unsigned __int8 *)a2, a3, a4);
      if ( result )
        return result;
      v18 = a4;
      v19 = a3;
      v20 = (unsigned __int8 *)a2;
      return sub_96F860(v20, v19, v18);
    default:
      BUG();
  }
}
