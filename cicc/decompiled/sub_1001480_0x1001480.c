// Function: sub_1001480
// Address: 0x1001480
//
__int64 __fastcall sub_1001480(unsigned int a1, unsigned __int8 *a2, __int64 a3, __int64 *a4)
{
  int v7; // eax
  __int64 v9; // r15
  __int64 v10; // rdx
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rcx
  _BYTE *v14; // rax
  unsigned __int8 *v15; // rdx
  int v16; // eax
  int v17; // eax
  unsigned __int8 *v18; // rdx
  unsigned int v19; // r10d
  int v20; // esi
  __int64 v21; // rcx
  __int64 v22; // r9
  int v23; // esi
  __int64 v24; // r8
  int v25; // esi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  unsigned int v32; // [rsp+1Ch] [rbp-44h]
  unsigned int v33; // [rsp+1Ch] [rbp-44h]
  unsigned int v34; // [rsp+1Ch] [rbp-44h]
  __int64 v35; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+20h] [rbp-40h]
  __int64 v37; // [rsp+20h] [rbp-40h]
  __int64 v38; // [rsp+28h] [rbp-38h]
  __int64 v39; // [rsp+28h] [rbp-38h]
  __int64 v40; // [rsp+28h] [rbp-38h]

  v7 = *a2;
  if ( (unsigned __int8)v7 <= 0x15u )
    return sub_96F480(a1, (__int64)a2, a3, *a4);
  if ( (unsigned __int8)v7 <= 0x1Cu )
    goto LABEL_6;
  if ( (unsigned int)(v7 - 67) > 0xC )
    goto LABEL_6;
  v9 = *((_QWORD *)a2 - 4);
  v10 = *(_QWORD *)(v9 + 8);
  if ( a3 != v10 )
    goto LABEL_6;
  v19 = v7 - 29;
  v20 = *(unsigned __int8 *)(a3 + 8);
  if ( (unsigned int)(v20 - 17) <= 1 )
    LOBYTE(v20) = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
  v21 = *((_QWORD *)a2 + 1);
  v22 = 0;
  if ( (_BYTE)v20 == 14 )
  {
    v33 = v7 - 29;
    v36 = *((_QWORD *)a2 + 1);
    v39 = *(_QWORD *)(v9 + 8);
    v28 = sub_AE4450(*a4, v39);
    v19 = v33;
    v21 = v36;
    v10 = v39;
    v22 = v28;
  }
  v23 = *(unsigned __int8 *)(v21 + 8);
  if ( (unsigned int)(v23 - 17) <= 1 )
    LOBYTE(v23) = *(_BYTE *)(**(_QWORD **)(v21 + 16) + 8LL);
  v24 = 0;
  if ( (_BYTE)v23 == 14 )
  {
    v30 = v22;
    v32 = v19;
    v35 = v10;
    v38 = v21;
    v27 = sub_AE4450(*a4, v21);
    v22 = v30;
    v19 = v32;
    v10 = v35;
    v21 = v38;
    v24 = v27;
  }
  v25 = *(unsigned __int8 *)(v10 + 8);
  if ( (unsigned int)(v25 - 17) <= 1 )
    LOBYTE(v25) = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
  v26 = 0;
  if ( (_BYTE)v25 == 14 )
  {
    v29 = v24;
    v31 = v22;
    v34 = v19;
    v37 = v21;
    v40 = v10;
    v26 = sub_AE4450(*a4, v10);
    v24 = v29;
    v22 = v31;
    v19 = v34;
    v21 = v37;
    v10 = v40;
  }
  if ( (unsigned int)sub_B50810(v19, a1, v10, v21, v10, v22, v24, v26) != 49 )
  {
LABEL_6:
    if ( a1 == 49 )
    {
      if ( a3 == *((_QWORD *)a2 + 1) )
        return (__int64)a2;
      return 0;
    }
    if ( a1 == 47 )
    {
      v11 = *a2;
      if ( *a2 <= 0x1Cu )
      {
        if ( v11 != 5 || *((_WORD *)a2 + 1) != 34 )
          return 0;
      }
      else if ( v11 != 63 )
      {
        return 0;
      }
      v12 = sub_BB5290((__int64)a2);
      if ( sub_BCAC40(v12, 8) )
      {
        v13 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        if ( v13 )
        {
          v14 = *(_BYTE **)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
          if ( *v14 == 44 )
          {
            v9 = *((_QWORD *)v14 - 8);
            if ( v9 )
            {
              v15 = (unsigned __int8 *)*((_QWORD *)v14 - 4);
              v16 = *v15;
              if ( (unsigned __int8)v16 > 0x1Cu )
              {
                v17 = v16 - 29;
                goto LABEL_17;
              }
              if ( (_BYTE)v16 == 5 )
              {
                v17 = *((unsigned __int16 *)v15 + 1);
LABEL_17:
                if ( v17 == 47 )
                {
                  v18 = (v15[7] & 0x40) != 0
                      ? (unsigned __int8 *)*((_QWORD *)v15 - 1)
                      : &v15[-32 * (*((_DWORD *)v15 + 1) & 0x7FFFFFF)];
                  if ( v13 == *(_QWORD *)v18 && a3 == *(_QWORD *)(v9 + 8) && a3 == sub_AE4570(*a4, *(_QWORD *)(v13 + 8)) )
                    return v9;
                }
              }
            }
          }
        }
      }
    }
    return 0;
  }
  return v9;
}
