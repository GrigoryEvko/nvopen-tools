// Function: sub_D11F30
// Address: 0xd11f30
//
__int64 __fastcall sub_D11F30(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v2; // r14
  __int64 *v3; // rbx
  __int64 v4; // r13
  __int64 v6; // rax
  bool v7; // zf
  _QWORD *v8; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v9[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v10; // [rsp+20h] [rbp-30h]
  __int64 v11; // [rsp+28h] [rbp-28h]

  v2 = (_QWORD *)sub_D110B0(a1, a2);
  if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1 || (unsigned __int8)sub_B2DDD0(a2, 0, 1, 1, 0, 0, 0) )
  {
    v3 = (__int64 *)a1[7];
    v11 = 0;
    v8 = v2;
    v4 = v3[3];
    if ( v4 == v3[4] )
    {
      sub_D10B90(v3 + 2, v3[3], (__int64)v9, &v8);
    }
    else
    {
      if ( v4 )
      {
        *(_BYTE *)(v4 + 24) = 0;
        if ( (_BYTE)v11 )
        {
          *(_QWORD *)v4 = 6;
          *(_QWORD *)(v4 + 8) = 0;
          v6 = v10;
          v7 = v10 == -4096;
          *(_QWORD *)(v4 + 16) = v10;
          if ( v6 != 0 && !v7 && v6 != -8192 )
            sub_BD6050((unsigned __int64 *)v4, v9[0] & 0xFFFFFFFFFFFFFFF8LL);
          *(_BYTE *)(v4 + 24) = 1;
        }
        *(_QWORD *)(v4 + 32) = v8;
        v4 = v3[3];
      }
      v3[3] = v4 + 40;
    }
    if ( (_BYTE)v11 )
    {
      LOBYTE(v11) = 0;
      if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        sub_BD60C0(v9);
    }
    ++*((_DWORD *)v8 + 10);
  }
  return sub_D11900(a1, v2);
}
