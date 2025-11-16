// Function: sub_153D2A0
// Address: 0x153d2a0
//
void __fastcall sub_153D2A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  char *v10; // r10
  char *v11; // r11
  char *v12; // r13
  __int64 v13; // rax
  __int64 *v14; // rax
  int v15; // edx
  char *v16; // [rsp+8h] [rbp-58h]
  __int64 **v17; // [rsp+10h] [rbp-50h]
  char *v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]
  __int64 v21[7]; // [rsp+28h] [rbp-38h] BYREF

  v21[0] = a6;
  if ( a4 && a5 )
  {
    if ( a4 + a5 == 2 )
    {
      if ( sub_153CA80((__int64)v21, (__int64 **)a2, (__int64 **)a1) )
      {
        v14 = *(__int64 **)a1;
        *(_QWORD *)a1 = *(_QWORD *)a2;
        v15 = *(_DWORD *)(a2 + 8);
        *(_QWORD *)a2 = v14;
        LODWORD(v14) = *(_DWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 8) = v15;
        *(_DWORD *)(a2 + 8) = (_DWORD)v14;
      }
    }
    else
    {
      if ( a4 > a5 )
      {
        v20 = a4 / 2;
        v13 = sub_153CE20(a2, a3, (__int64 **)(a1 + 16 * (a4 / 2)), v21[0]);
        v11 = (char *)(a1 + 16 * (a4 / 2));
        v10 = (char *)v13;
        v19 = (v13 - a2) >> 4;
      }
      else
      {
        v19 = a5 / 2;
        v17 = (__int64 **)(a2 + 16 * (a5 / 2));
        v9 = sub_153CEB0(a1, a2, v17, v21[0]);
        v10 = (char *)v17;
        v11 = (char *)v9;
        v20 = (v9 - a1) >> 4;
      }
      v16 = v10;
      v18 = v11;
      v12 = sub_153C900(v11, (char *)a2, v10);
      sub_153D2A0(a1, v18, v12, v20, v19, v21[0]);
      sub_153D2A0(v12, v16, a3, a4 - v20, a5 - v19, v21[0]);
    }
  }
}
