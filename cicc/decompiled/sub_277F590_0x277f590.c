// Function: sub_277F590
// Address: 0x277f590
//
unsigned __int64 __fastcall sub_277F590(unsigned __int8 *a1)
{
  int v2; // ebx
  __int64 v3; // r13
  unsigned __int64 v4; // r14
  char v5; // al
  __int64 v7; // rax
  __int16 v8; // di
  unsigned int v9; // eax
  __int64 v10; // rcx
  unsigned __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // ebx
  int v16; // r15d
  int v17; // eax
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rax
  __int64 *v22; // rsi
  unsigned __int8 *v23; // rdx
  __int64 *v24; // rsi
  __int64 *v25; // rdi
  int v26; // [rsp+8h] [rbp-68h] BYREF
  int v27; // [rsp+Ch] [rbp-64h] BYREF
  _BYTE *v28; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 v29; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v30; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v31; // [rsp+28h] [rbp-48h] BYREF
  unsigned __int64 v32; // [rsp+30h] [rbp-40h] BYREF
  __int64 v33[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = *a1;
  if ( (unsigned int)(v2 - 42) > 0x11 )
  {
    if ( (unsigned __int8)(v2 - 82) > 1u )
    {
      if ( (unsigned __int8)sub_2779B30((__int64)a1, (__int64 *)&v28, &v29, &v30, &v26) )
      {
        if ( (unsigned int)(v26 - 1) > 3 )
        {
          if ( (unsigned __int8)(*v28 - 82) <= 1u
            && *((_QWORD *)v28 - 8)
            && (v32 = *((_QWORD *)v28 - 8), *((_QWORD *)v28 - 4)) )
          {
            v33[0] = *((_QWORD *)v28 - 4);
            v15 = sub_B53900((__int64)v28);
            v16 = v15;
            if ( (unsigned int)sub_B52870(v15) < v15 )
            {
              v16 = sub_B52870(v15);
              v18 = v29;
              v29 = v30;
              v30 = v18;
            }
            v17 = *a1;
            LODWORD(v31) = v16;
            v27 = v17 - 29;
            return sub_277EFE0(&v27, (int *)&v31, (__int64 *)&v32, v33, (__int64 *)&v29, (__int64 *)&v30);
          }
          else
          {
            LODWORD(v33[0]) = *a1 - 29;
            return sub_277EC00((int *)v33, (__int64 *)&v28, (__int64 *)&v29, (__int64 *)&v30);
          }
        }
        else
        {
          v11 = v29;
          if ( v29 > v30 )
          {
            v29 = v30;
            v30 = v11;
          }
          LODWORD(v33[0]) = *a1 - 29;
          return sub_277E440((int *)v33, &v26, (__int64 *)&v29, (__int64 *)&v30);
        }
      }
      else
      {
        v12 = *a1;
        if ( (unsigned int)(v12 - 67) > 0xC )
        {
          switch ( (_BYTE)v12 )
          {
            case '`':
              v14 = *((_QWORD *)a1 - 4);
              LODWORD(v32) = 67;
              v33[0] = v14;
              return sub_277C570((int *)&v32, v33);
            case ']':
              v32 = sub_939680(*((_QWORD **)a1 + 9), *((_QWORD *)a1 + 9) + 4LL * *((unsigned int *)a1 + 20));
              v33[0] = *((_QWORD *)a1 - 4);
              LODWORD(v31) = *a1 - 29;
              return sub_277C1C0((int *)&v31, v33, (__int64 *)&v32);
            case '^':
              v31 = sub_939680(*((_QWORD **)a1 + 9), *((_QWORD *)a1 + 9) + 4LL * *((unsigned int *)a1 + 20));
              v33[0] = *((_QWORD *)a1 - 4);
              v32 = *((_QWORD *)a1 - 8);
              v27 = *a1 - 29;
              return sub_277DE70(&v27, (__int64 *)&v32, v33, (__int64 *)&v31);
            default:
              if ( sub_988010((__int64)a1) && sub_277ABC0((__int64)a1) && (unsigned int)sub_A17190(a1) > 1 )
              {
                v19 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
                v20 = *(_QWORD *)&a1[-32 * v19];
                v31 = v20;
                v21 = *(_QWORD *)&a1[32 * (1 - v19)];
                v32 = v21;
                if ( v21 < v20 )
                {
                  v31 = v21;
                  v32 = v20;
                }
                v22 = (__int64 *)sub_277ABA0((__int64)a1);
                if ( (a1[7] & 0x40) != 0 )
                  v23 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
                else
                  v23 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
                v33[0] = sub_F58E90((__int64 *)v23 + 8, v22);
                v27 = *a1 - 29;
                return sub_277DE70(&v27, (__int64 *)&v31, (__int64 *)&v32, v33);
              }
              else if ( sub_277B110((__int64)a1) )
              {
                v33[0] = sub_B5B890((__int64)a1);
                v32 = sub_B5B740((__int64)a1);
                v31 = *(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
                v27 = *a1 - 29;
                return sub_277EC00(&v27, (__int64 *)&v31, (__int64 *)&v32, v33);
              }
              else if ( *a1 == 85 )
              {
                return sub_277CF80((__int64 *)a1);
              }
              else
              {
                v24 = (__int64 *)sub_277ABA0((__int64)a1);
                if ( (a1[7] & 0x40) != 0 )
                  v25 = (__int64 *)*((_QWORD *)a1 - 1);
                else
                  v25 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
                v33[0] = sub_F58E90(v25, v24);
                LODWORD(v32) = *a1 - 29;
                return sub_C4ECF0((int *)&v32, v33);
              }
          }
        }
        else
        {
          v13 = *((_QWORD *)a1 - 4);
          LODWORD(v31) = v12 - 29;
          v33[0] = v13;
          v32 = *((_QWORD *)a1 + 1);
          return sub_277E250((int *)&v31, (__int64 *)&v32, v33);
        }
      }
    }
    else
    {
      v32 = *((_QWORD *)a1 - 8);
      v7 = *((_QWORD *)a1 - 4);
      v8 = *((_WORD *)a1 + 1);
      v33[0] = v7;
      LODWORD(v30) = v8 & 0x3F;
      v9 = sub_B52F50(v30);
      v10 = v32;
      if ( v33[0] < v32 || v33[0] == v32 && v9 < (unsigned int)v30 )
      {
        v32 = v33[0];
        v33[0] = v10;
        LODWORD(v30) = v9;
      }
      LODWORD(v31) = *a1 - 29;
      return sub_277E820((int *)&v31, (int *)&v30, (__int64 *)&v32, v33);
    }
  }
  else
  {
    v4 = *((_QWORD *)a1 - 4);
    v32 = *((_QWORD *)a1 - 8);
    v3 = v32;
    v33[0] = v4;
    v5 = sub_B46D50(a1);
    if ( v4 < v32 )
    {
      if ( v5 )
      {
        v32 = v4;
        v33[0] = v3;
      }
    }
    LODWORD(v31) = v2 - 29;
    return sub_277F3A0((int *)&v31, (__int64 *)&v32, v33);
  }
}
