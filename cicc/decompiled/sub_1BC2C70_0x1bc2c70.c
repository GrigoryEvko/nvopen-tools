// Function: sub_1BC2C70
// Address: 0x1bc2c70
//
__int64 __fastcall sub_1BC2C70(__int64 a1, __int64 a2)
{
  int **v3; // rdx
  int **v4; // rbx
  int **v7; // rsi
  __int64 v8; // rdi
  int **v9; // rax
  int v10; // r8d
  __int64 v11; // rax
  int *v12; // rdx
  _QWORD v13[2]; // [rsp+0h] [rbp-80h] BYREF
  int **v14; // [rsp+10h] [rbp-70h]
  int **v15; // [rsp+18h] [rbp-68h]
  int v16; // [rsp+30h] [rbp-50h]
  int v17; // [rsp+50h] [rbp-30h]

  v3 = *(int ***)(a2 + 1272);
  v4 = &v3[5 * *(unsigned int *)(a2 + 1288)];
  if ( !*(_DWORD *)(a2 + 1280) )
    goto LABEL_2;
  v13[1] = *(_QWORD *)(a2 + 1264);
  v13[0] = a2 + 1264;
  v14 = v3;
  v15 = v4;
  sub_1BC2C00((__int64)v13);
  v7 = v14;
  v8 = *(_QWORD *)(a2 + 1272) + 40LL * *(unsigned int *)(a2 + 1288);
  if ( v4 != v14 )
  {
LABEL_4:
    v9 = v7;
    while ( 1 )
    {
      v9 += 5;
      v16 = -2;
      v17 = -3;
      if ( v9 != v15 )
      {
        while ( *((_DWORD *)v9 + 2) == 1 )
        {
          v10 = **v9;
          if ( v16 != v10 && v17 != v10 )
            break;
          v9 += 5;
          if ( v9 == v15 )
          {
            if ( v4 != v9 )
              goto LABEL_6;
            goto LABEL_13;
          }
        }
      }
      if ( v4 == v9 )
        break;
LABEL_6:
      if ( *((_DWORD *)v7 + 8) < *((_DWORD *)v9 + 8) )
      {
        v7 = v9;
        goto LABEL_4;
      }
    }
  }
LABEL_13:
  if ( (int **)v8 == v7 || *((_DWORD *)v7 + 8) <= *(_DWORD *)(a2 + 1296) )
  {
LABEL_2:
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  else
  {
    v11 = *((unsigned int *)v7 + 2);
    v12 = *v7;
    *(_BYTE *)(a1 + 16) = 1;
    *(_QWORD *)(a1 + 8) = v11;
    *(_QWORD *)a1 = v12;
    return a1;
  }
}
