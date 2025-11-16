// Function: sub_1758400
// Address: 0x1758400
//
_QWORD *__fastcall sub_1758400(__int64 a1, __int64 a2, _QWORD ***a3, __int64 a4, double a5, double a6, double a7)
{
  int v9; // ebx
  unsigned int v10; // eax
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 *v13; // r15
  __int64 v14; // rbx
  unsigned int v15; // eax
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // rbx
  _QWORD *v19; // r12
  _QWORD **v20; // rax
  _QWORD *v21; // r15
  __int64 *v22; // rax
  __int64 v23; // rsi
  __int64 v25; // rbx
  _QWORD **v26; // rax
  _QWORD *v27; // r14
  __int64 *v28; // rax
  __int64 v29; // rsi
  __int64 *v30; // rax
  __int64 v31; // rbx
  _QWORD **v32; // rax
  _QWORD *v33; // r14
  __int64 *v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rbx
  _QWORD **v38; // rax
  _QWORD *v39; // r15
  __int64 *v40; // rax
  __int64 v41; // rsi
  unsigned int v42; // [rsp+4h] [rbp-6Ch]
  _QWORD **v43; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v44; // [rsp+18h] [rbp-58h]
  unsigned __int64 v45; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v46; // [rsp+28h] [rbp-48h]
  __int16 v47; // [rsp+30h] [rbp-40h]

  if ( (unsigned int)(a4 - 36) <= 1 )
  {
    v30 = (__int64 *)sub_15A04A0(*a3);
    v47 = 257;
    v31 = sub_15A2B60(v30, (__int64)a3, 0, 0, a5, a6, a7);
    v19 = sub_1648A60(56, 2u);
    if ( v19 )
    {
      v32 = *(_QWORD ***)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      {
        v33 = v32[4];
        v34 = (__int64 *)sub_1643320(*v32);
        v35 = (__int64)sub_16463B0(v34, (unsigned int)v33);
      }
      else
      {
        v35 = sub_1643320(*v32);
      }
      sub_15FEC10((__int64)v19, v35, 51, 34, a2, v31, (__int64)&v45, 0);
    }
  }
  else
  {
    v9 = a4;
    if ( (unsigned int)(a4 - 34) > 1 )
    {
      v10 = sub_1643030((__int64)*a3);
      v46 = v10;
      v11 = ~(1LL << ((unsigned __int8)v10 - 1));
      if ( v10 > 0x40 )
      {
        v42 = v10 - 1;
        sub_16A4EF0((__int64)&v45, -1, 1);
        if ( v46 > 0x40 )
        {
          *(_QWORD *)(v45 + 8LL * (v42 >> 6)) &= v11;
LABEL_6:
          v12 = (__int64 *)sub_16498A0(a2);
          v13 = (__int64 *)sub_159C0E0(v12, (__int64)&v45);
          if ( v46 > 0x40 && v45 )
            j_j___libc_free_0_0(v45);
          if ( (unsigned int)(v9 - 40) <= 1 )
          {
            v36 = sub_15A2B60(v13, (__int64)a3, 0, 0, a5, a6, a7);
            v47 = 257;
            v37 = v36;
            v19 = sub_1648A60(56, 2u);
            if ( v19 )
            {
              v38 = *(_QWORD ***)a2;
              if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
              {
                v39 = v38[4];
                v40 = (__int64 *)sub_1643320(*v38);
                v41 = (__int64)sub_16463B0(v40, (unsigned int)v39);
              }
              else
              {
                v41 = sub_1643320(*v38);
              }
              sub_15FEC10((__int64)v19, v41, 51, 38, a2, v37, (__int64)&v45, 0);
            }
          }
          else
          {
            v14 = *(_QWORD *)(a1 + 8);
            v44 = *((_DWORD *)a3 + 8);
            if ( v44 > 0x40 )
              sub_16A4FD0((__int64)&v43, (const void **)a3 + 3);
            else
              v43 = a3[3];
            sub_16A7800((__int64)&v43, 1u);
            v15 = v44;
            v44 = 0;
            v46 = v15;
            v45 = (unsigned __int64)v43;
            v16 = sub_159C0E0(*(__int64 **)(v14 + 24), (__int64)&v45);
            if ( v46 > 0x40 && v45 )
              j_j___libc_free_0_0(v45);
            if ( v44 > 0x40 && v43 )
              j_j___libc_free_0_0(v43);
            v17 = sub_15A2B60(v13, v16, 0, 0, a5, a6, a7);
            v47 = 257;
            v18 = v17;
            v19 = sub_1648A60(56, 2u);
            if ( v19 )
            {
              v20 = *(_QWORD ***)a2;
              if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
              {
                v21 = v20[4];
                v22 = (__int64 *)sub_1643320(*v20);
                v23 = (__int64)sub_16463B0(v22, (unsigned int)v21);
              }
              else
              {
                v23 = sub_1643320(*v20);
              }
              sub_15FEC10((__int64)v19, v23, 51, 40, a2, v18, (__int64)&v45, 0);
            }
          }
          return v19;
        }
      }
      else
      {
        v45 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v10;
      }
      v45 &= v11;
      goto LABEL_6;
    }
    v47 = 257;
    v25 = sub_15A2B90((__int64 *)a3, 0, 0, a4, a5, a6, a7);
    v19 = sub_1648A60(56, 2u);
    if ( v19 )
    {
      v26 = *(_QWORD ***)a2;
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      {
        v27 = v26[4];
        v28 = (__int64 *)sub_1643320(*v26);
        v29 = (__int64)sub_16463B0(v28, (unsigned int)v27);
      }
      else
      {
        v29 = sub_1643320(*v26);
      }
      sub_15FEC10((__int64)v19, v29, 51, 36, a2, v25, (__int64)&v45, 0);
    }
  }
  return v19;
}
