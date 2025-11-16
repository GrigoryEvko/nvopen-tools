// Function: sub_1DE4C10
// Address: 0x1de4c10
//
void __fastcall sub_1DE4C10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // r11
  __int64 v7; // r9
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 *v10; // r9
  __int64 *v11; // r10
  __int64 v12; // r11
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 *v15; // rax
  unsigned int v16; // edx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 *v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  __int64 *v22; // [rsp+18h] [rbp-38h]

  if ( a4 )
  {
    v5 = a5;
    if ( a5 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a4;
      if ( a4 + a5 == 2 )
      {
        v13 = a2;
        v15 = (__int64 *)a1;
LABEL_12:
        v16 = *((_DWORD *)v15 + 2);
        if ( v16 < *(_DWORD *)(v13 + 8) )
        {
          *((_DWORD *)v15 + 2) = *(_DWORD *)(v13 + 8);
          v17 = *(_QWORD *)v13;
          *(_DWORD *)(v13 + 8) = v16;
          v18 = *v15;
          *v15 = v17;
          *(_QWORD *)v13 = v18;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v9 = v8 / 2;
        v13 = sub_1DE41C0(v7, a3, v6 + 16 * (v8 / 2));
        v14 = (v13 - (__int64)v10) >> 4;
        while ( 1 )
        {
          v21 = v12;
          v22 = v11;
          v5 -= v14;
          v20 = sub_1DE36B0(v11, v10, (__int64 *)v13);
          sub_1DE4C10(v21, v22, v20, v9, v14);
          v8 -= v9;
          if ( !v8 )
            break;
          v15 = v20;
          if ( !v5 )
            break;
          if ( v5 + v8 == 2 )
            goto LABEL_12;
          v7 = v13;
          v6 = (__int64)v20;
          if ( v8 > v5 )
            goto LABEL_5;
LABEL_10:
          v14 = v5 / 2;
          v13 = v7 + 16 * (v5 / 2);
          v11 = (__int64 *)sub_1DE4210(v6, v7, v13);
          v9 = ((__int64)v11 - v12) >> 4;
        }
      }
    }
  }
}
