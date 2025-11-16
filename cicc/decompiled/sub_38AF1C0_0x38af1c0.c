// Function: sub_38AF1C0
// Address: 0x38af1c0
//
__int64 __fastcall sub_38AF1C0(__int64 a1, _QWORD *a2, __int64 *a3, int a4, double a5, double a6, double a7)
{
  unsigned int v9; // r13d
  unsigned __int64 v11; // r14
  __int64 v12; // rdx
  char v13; // al
  _QWORD *v14; // rbx
  __int64 v15; // r14
  __int16 v16; // r12
  _QWORD **v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r9
  const char *v21; // rax
  __int64 v22; // r14
  __int16 v23; // r12
  _QWORD **v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rax
  __int64 v27; // r9
  __int64 v28; // [rsp+0h] [rbp-80h]
  __int64 v29; // [rsp+0h] [rbp-80h]
  _QWORD *v30; // [rsp+8h] [rbp-78h]
  _QWORD *v31; // [rsp+8h] [rbp-78h]
  __int64 v32; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+8h] [rbp-78h]
  int v34; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 *v35; // [rsp+20h] [rbp-60h] BYREF
  __int64 v36; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v37[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v38; // [rsp+40h] [rbp-40h]

  if ( !(unsigned __int8)sub_388E880(a1, &v34, a4) )
  {
    v11 = *(_QWORD *)(a1 + 56);
    if ( !(unsigned __int8)sub_38AB270((__int64 **)a1, &v35, a3, a5, a6, a7)
      && !(unsigned __int8)sub_388AF10(a1, 4, "expected ',' after compare value") )
    {
      v9 = sub_38A1070((__int64 **)a1, *v35, &v36, a3, a5, a6, a7);
      if ( !(_BYTE)v9 )
      {
        v12 = *v35;
        v13 = *(_BYTE *)(*v35 + 8);
        if ( a4 == 52 )
        {
          if ( v13 == 16 )
            v13 = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
          if ( (unsigned __int8)(v13 - 1) <= 5u )
          {
            v38 = 257;
            v14 = sub_1648A60(56, 2u);
            if ( v14 )
            {
              v22 = (__int64)v35;
              v23 = v34;
              v24 = (_QWORD **)*v35;
              if ( *(_BYTE *)(*v35 + 8) == 16 )
              {
                v29 = v36;
                v31 = v24[4];
                v25 = (__int64 *)sub_1643320(*v24);
                v26 = (__int64)sub_16463B0(v25, (unsigned int)v31);
                v27 = v29;
              }
              else
              {
                v33 = v36;
                v26 = sub_1643320(*v24);
                v27 = v33;
              }
              sub_15FEC10((__int64)v14, v26, 52, v23, v22, v27, (__int64)v37, 0);
            }
            goto LABEL_15;
          }
          HIBYTE(v38) = 1;
          v21 = "fcmp requires floating point operands";
        }
        else
        {
          if ( v13 == 16 )
            v13 = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
          if ( (v13 & 0xFB) == 0xB )
          {
            v38 = 257;
            v14 = sub_1648A60(56, 2u);
            if ( v14 )
            {
              v15 = (__int64)v35;
              v16 = v34;
              v17 = (_QWORD **)*v35;
              if ( *(_BYTE *)(*v35 + 8) == 16 )
              {
                v28 = v36;
                v30 = v17[4];
                v18 = (__int64 *)sub_1643320(*v17);
                v19 = (__int64)sub_16463B0(v18, (unsigned int)v30);
                v20 = v28;
              }
              else
              {
                v32 = v36;
                v19 = sub_1643320(*v17);
                v20 = v32;
              }
              sub_15FEC10((__int64)v14, v19, 51, v16, v15, v20, (__int64)v37, 0);
            }
LABEL_15:
            *a2 = v14;
            return v9;
          }
          HIBYTE(v38) = 1;
          v21 = "icmp requires integer operands";
        }
        v37[0] = v21;
        LOBYTE(v38) = 3;
        return (unsigned int)sub_38814C0(a1 + 8, v11, (__int64)v37);
      }
    }
  }
  return 1;
}
