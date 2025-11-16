// Function: sub_2B12D30
// Address: 0x2b12d30
//
void __fastcall sub_2B12D30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r11
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r9
  __int64 v11; // r10
  __int64 v12; // r11
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // ecx
  int v18; // ecx
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  __int64 v22; // [rsp+18h] [rbp-38h]

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a5;
      if ( a4 + a5 == 2 )
      {
        v15 = a1;
        v13 = a2;
LABEL_12:
        if ( *(_DWORD *)(v13 + 4) < *(_DWORD *)(v15 + 4) )
        {
          v16 = *(_QWORD *)(v15 + 8);
          *(_QWORD *)(v15 + 8) = *(_QWORD *)(v13 + 8);
          v17 = *(_DWORD *)(v13 + 4);
          *(_QWORD *)(v13 + 8) = v16;
          LODWORD(v16) = *(_DWORD *)(v15 + 4);
          *(_DWORD *)(v15 + 4) = v17;
          v18 = *(_DWORD *)v13;
          *(_DWORD *)(v13 + 4) = v16;
          LODWORD(v16) = *(_DWORD *)v15;
          *(_DWORD *)v15 = v18;
          *(_DWORD *)v13 = v16;
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v13 = sub_2B0EF00(v7, a3, v6 + 16 * (v5 / 2));
        v14 = (v13 - v10) >> 4;
        while ( 1 )
        {
          v21 = v12;
          v22 = v11;
          v8 -= v14;
          v20 = sub_2B0A8B0(v11, v10, v13);
          sub_2B12D30(v21, v22, v20, v9, v14);
          v5 -= v9;
          if ( !v5 )
            break;
          v15 = v20;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v6 = v20;
          v7 = v13;
          if ( v8 < v5 )
            goto LABEL_5;
LABEL_10:
          v14 = v8 / 2;
          v13 = v7 + 16 * (v8 / 2);
          v11 = sub_2B0EF50(v6, v7, v13);
          v9 = (v11 - v12) >> 4;
        }
      }
    }
  }
}
