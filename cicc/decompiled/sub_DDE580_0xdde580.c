// Function: sub_DDE580
// Address: 0xdde580
//
__int64 __fastcall sub_DDE580(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  _BYTE *v9; // r13
  bool v12; // al
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned int v18; // eax
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  char v22; // al
  __int64 v23; // r11
  __int64 *v24; // r10
  bool v25; // al
  _BYTE *v26; // rcx
  unsigned int v27; // edi
  _BYTE *v28; // rbx
  unsigned int v29; // eax
  __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 *v32; // [rsp+8h] [rbp-58h]
  _BYTE *v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 *v35; // [rsp+10h] [rbp-50h]
  unsigned __int64 v36; // [rsp+18h] [rbp-48h]
  unsigned int v37; // [rsp+24h] [rbp-3Ch]
  _QWORD *v39; // [rsp+28h] [rbp-38h]
  __int64 v40; // [rsp+28h] [rbp-38h]
  _QWORD *v41; // [rsp+28h] [rbp-38h]
  __int64 v42; // [rsp+28h] [rbp-38h]

  v9 = (_BYTE *)a5;
  v37 = a3;
  v36 = HIDWORD(a3);
  v12 = sub_DADE90((__int64)a2, a5, a6);
  v16 = a6;
  if ( !v12 )
  {
    if ( !sub_DADE90((__int64)a2, a4, a6) )
      goto LABEL_4;
    v18 = sub_B52F50(a3);
    v16 = a6;
    v37 = v18;
    v19 = (_BYTE *)a4;
    a4 = (__int64)v9;
    v9 = v19;
  }
  if ( *(_WORD *)(a4 + 24) == 8 && v16 == *(_QWORD *)(a4 + 48) && v37 - 32 > 1 )
  {
    v31 = v16;
    v34 = sub_D33D80((_QWORD *)a4, (__int64)a2, v13, v14, v15);
    v20 = sub_D95540(v34);
    v39 = sub_DA2C50((__int64)a2, v20, 1, 0);
    v21 = sub_DCAF50(a2, (__int64)v39, 0);
    if ( (_QWORD *)v34 == v39 || (__int64 *)v34 == v21 )
    {
      v30 = v31;
      v32 = v21;
      v40 = sub_D95540(**(_QWORD **)(a4 + 32));
      if ( v40 == sub_D95540(a8) )
      {
        v41 = sub_DD0540(a4, a8, a2);
        v22 = sub_DDDA00((__int64)a2, v30, a3 & 0xFFFFFFFF00000000LL | v37, (__int64)v41, v9);
        v23 = v34;
        v24 = v32;
        if ( v22 )
        {
          v33 = v41;
          v35 = v24;
          v42 = v23;
          v25 = sub_B532B0(v37);
          v26 = v33;
          v27 = !v25 ? 37 : 41;
          if ( (__int64 *)v42 == v35 )
          {
            v29 = sub_B52F50(v27);
            v26 = v33;
            v27 = v29;
          }
          v28 = **(_BYTE ***)(a4 + 32);
          if ( (unsigned __int8)sub_DDCB50(a2, v27, v28, v26, a7) )
          {
            *(_QWORD *)(a1 + 8) = v28;
            *(_QWORD *)(a1 + 16) = v9;
            *(_DWORD *)a1 = v37;
            *(_BYTE *)(a1 + 24) = 1;
            *(_BYTE *)(a1 + 4) = v36;
            return a1;
          }
        }
      }
    }
  }
LABEL_4:
  *(_BYTE *)(a1 + 24) = 0;
  return a1;
}
