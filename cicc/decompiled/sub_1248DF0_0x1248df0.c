// Function: sub_1248DF0
// Address: 0x1248df0
//
__int64 __fastcall sub_1248DF0(__int64 a1, __int64 a2, unsigned int a3, int *a4)
{
  __int64 v4; // r13
  unsigned __int64 v5; // rsi
  unsigned int v6; // r12d
  int v8; // eax
  __int64 v9; // r14
  unsigned __int64 v10; // rsi
  int v11; // eax
  const char *v14; // [rsp+10h] [rbp-100h] BYREF
  char v15; // [rsp+30h] [rbp-E0h]
  char v16; // [rsp+31h] [rbp-DFh]
  __int64 v17[4]; // [rsp+40h] [rbp-D0h] BYREF
  char v18; // [rsp+60h] [rbp-B0h]
  char v19; // [rsp+61h] [rbp-AFh]

  v4 = a1 + 176;
  if ( *(_DWORD *)(a1 + 240) == 8 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    sub_1247D30((__int64)v17, a1, a2, a3, a4);
    v6 = sub_121E130((__int64)v17);
    if ( !(_BYTE)v6 )
    {
      v8 = *(_DWORD *)(a1 + 240);
      v9 = *(_QWORD *)(a1 + 1328);
      *(_QWORD *)(a1 + 1328) = v17;
      LOBYTE(v6) = v8 == 407 || v8 == 9;
      if ( (_BYTE)v6 )
      {
        v10 = *(_QWORD *)(a1 + 232);
        v16 = 1;
        v14 = "function body requires at least one basic block";
        v15 = 3;
        sub_11FD800(v4, v10, (__int64)&v14, 1);
      }
      else
      {
        while ( !(unsigned __int8)sub_1248770(a1, v17) )
        {
          v11 = *(_DWORD *)(a1 + 240);
          if ( v11 == 9 )
            goto LABEL_12;
          if ( v11 == 407 )
          {
            while ( v11 != 9 )
            {
              if ( (unsigned __int8)sub_1240EC0(a1, v17) )
                goto LABEL_16;
              v11 = *(_DWORD *)(a1 + 240);
            }
LABEL_12:
            *(_DWORD *)(a1 + 240) = sub_1205200(v4);
            v6 = sub_1212490(v17);
            goto LABEL_7;
          }
        }
LABEL_16:
        v6 = 1;
      }
LABEL_7:
      *(_QWORD *)(a1 + 1328) = v9;
    }
    sub_120EAC0((__int64)v17);
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 232);
    v19 = 1;
    v6 = 1;
    v17[0] = (__int64)"expected '{' in function body";
    v18 = 3;
    sub_11FD800(a1 + 176, v5, (__int64)v17, 1);
  }
  return v6;
}
