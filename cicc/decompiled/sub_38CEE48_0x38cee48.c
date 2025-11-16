// Function: sub_38CEE48
// Address: 0x38cee48
//
void __fastcall sub_38CEE48(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // rbp
  __int64 v8; // r10
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 *v12; // r15
  int v13; // eax
  __int64 v14; // rdi
  int v15; // esi
  __int64 v16; // r10
  __int64 v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx

  v13 = *(_DWORD *)(v7 - 132);
  v14 = *(_QWORD *)(v6 + 32);
  *(_QWORD *)(v7 - 120) = v8;
  if ( (unsigned __int8)sub_38CEAE0(v14, v7 - 80, v11, v10, a5, v8, v13) )
  {
    v15 = *(_DWORD *)(v6 + 16);
    v16 = *(_QWORD *)(v7 - 120);
    if ( *(_QWORD *)(v7 - 112) || *(_QWORD *)(v7 - 104) || *(_QWORD *)(v7 - 80) || *(_QWORD *)(v7 - 72) )
    {
      if ( v15 )
      {
        if ( v15 == 17 )
          sub_38CE940(
            v11,
            v10,
            v16,
            *(_DWORD *)(v7 - 132),
            v12,
            *(_QWORD *)(v7 - 72),
            *(_QWORD *)(v7 - 80),
            -*(_QWORD *)(v7 - 64),
            (__int64 *)v9);
      }
      else
      {
        sub_38CE940(
          v11,
          v10,
          v16,
          *(_DWORD *)(v7 - 132),
          v12,
          *(_QWORD *)(v7 - 80),
          *(_QWORD *)(v7 - 72),
          *(_QWORD *)(v7 - 64),
          (__int64 *)v9);
      }
    }
    else
    {
      v17 = *(_QWORD *)(v7 - 96);
      v18 = *(_QWORD *)(v7 - 64);
      switch ( v15 )
      {
        case 0:
          v5 = v18 + v17;
          goto LABEL_14;
        case 1:
          v5 = v18 & v17;
          goto LABEL_14;
        case 2:
        case 10:
          if ( !v18 )
            break;
          v19 = v17 % v18;
          v5 = v17 / v18;
          if ( v15 != 2 && (v5 = v19, ((1LL << v15) & 0x1338) != 0) )
          {
LABEL_20:
            *(_QWORD *)v9 = 0;
            *(_QWORD *)(v9 + 8) = 0;
            *(_DWORD *)(v9 + 24) = 0;
            *(_QWORD *)(v9 + 16) = -(__int64)(v5 != 0);
          }
          else
          {
LABEL_14:
            *(_QWORD *)v9 = 0;
            *(_QWORD *)(v9 + 8) = 0;
            *(_QWORD *)(v9 + 16) = v5;
            *(_DWORD *)(v9 + 24) = 0;
          }
          break;
        case 3:
          v5 = v18 == v17;
          goto LABEL_20;
        case 4:
          v5 = v18 < v17;
          goto LABEL_20;
        case 5:
          v5 = v18 <= v17;
          goto LABEL_20;
        case 6:
          v5 = (v17 != 0) & (unsigned __int8)(v18 != 0);
          goto LABEL_14;
        case 7:
          v5 = (v17 | v18) != 0;
          goto LABEL_14;
        case 8:
          v5 = v18 > v17;
          goto LABEL_20;
        case 9:
          v5 = v18 >= v17;
          goto LABEL_20;
        case 11:
          v5 = v18 * v17;
          goto LABEL_14;
        case 12:
          v5 = v18 != v17;
          goto LABEL_20;
        case 13:
          v5 = v18 | v17;
          goto LABEL_14;
        case 14:
          v5 = v17 << v18;
          goto LABEL_14;
        case 15:
          v5 = v17 >> v18;
          goto LABEL_14;
        case 16:
          v5 = (unsigned __int64)v17 >> v18;
          goto LABEL_14;
        case 17:
          v5 = v17 - v18;
          goto LABEL_14;
        case 18:
          v5 = v18 ^ v17;
          goto LABEL_14;
        default:
          v5 = 0;
          goto LABEL_14;
      }
    }
    JUMPOUT(0x38CED29);
  }
  JUMPOUT(0x38CEC7C);
}
