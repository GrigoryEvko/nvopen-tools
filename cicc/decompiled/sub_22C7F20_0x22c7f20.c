// Function: sub_22C7F20
// Address: 0x22c7f20
//
__int64 __fastcall sub_22C7F20(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v7; // al
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned __int8 v10[48]; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int8 v11[8]; // [rsp+40h] [rbp-90h] BYREF
  const void *v12; // [rsp+48h] [rbp-88h] BYREF
  unsigned int v13; // [rsp+50h] [rbp-80h]
  const void *v14; // [rsp+58h] [rbp-78h] BYREF
  unsigned int v15; // [rsp+60h] [rbp-70h]
  char v16; // [rsp+68h] [rbp-68h]
  unsigned __int8 v17[40]; // [rsp+70h] [rbp-60h] BYREF
  char v18; // [rsp+98h] [rbp-38h]

  sub_22C7100((__int64)v11, a2, *(_QWORD *)(a3 - 64), a4, a3);
  if ( v16 )
  {
    sub_22C7100((__int64)v17, a2, *(_QWORD *)(a3 - 96), a4, a3);
    if ( v18 )
    {
      if ( v11[0] == 2 )
      {
        *(_BYTE *)(a1 + 40) = 1;
        *(_WORD *)a1 = 6;
        *(_WORD *)v10 = 0;
        sub_22C0090(v10);
      }
      else
      {
        sub_22C0C70((__int64)v11, (__int64)v17, 0, 0, 1u);
        v7 = v11[0];
        *(_BYTE *)(a1 + 1) = 0;
        *(_BYTE *)a1 = v7;
        if ( v7 > 3u )
        {
          if ( (unsigned __int8)(v7 - 4) <= 1u )
          {
            v8 = v13;
            *(_DWORD *)(a1 + 16) = v13;
            if ( v8 > 0x40 )
              sub_C43780(a1 + 8, &v12);
            else
              *(_QWORD *)(a1 + 8) = v12;
            v9 = v15;
            *(_DWORD *)(a1 + 32) = v15;
            if ( v9 > 0x40 )
              sub_C43780(a1 + 24, &v14);
            else
              *(_QWORD *)(a1 + 24) = v14;
            *(_BYTE *)(a1 + 1) = v11[1];
          }
        }
        else if ( v7 > 1u )
        {
          *(_QWORD *)(a1 + 8) = v12;
        }
        *(_BYTE *)(a1 + 40) = 1;
      }
      if ( v18 )
      {
        v18 = 0;
        sub_22C0090(v17);
      }
    }
    else
    {
      *(_BYTE *)(a1 + 40) = 0;
    }
    if ( v16 )
    {
      v16 = 0;
      sub_22C0090(v11);
    }
  }
  else
  {
    *(_BYTE *)(a1 + 40) = 0;
  }
  return a1;
}
