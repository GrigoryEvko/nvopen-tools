// Function: sub_2E868D0
// Address: 0x2e868d0
//
unsigned __int64 __fastcall sub_2E868D0(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  int *v3; // r8
  int v4; // eax
  __int64 v5; // r10
  unsigned __int8 v6; // cl
  unsigned __int8 v7; // r9
  unsigned __int8 v8; // r11
  unsigned __int8 v9; // bl
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // r11
  __int64 v13; // r9

  result = *(_QWORD *)(a1 + 48);
  v3 = (int *)(result & 0xFFFFFFFFFFFFFFF8LL);
  if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (result & 7) != 0 )
    {
      if ( (result & 7) != 3 || !*v3 )
        return result;
    }
    else
    {
      *(_QWORD *)(a1 + 48) = v3;
      LOBYTE(result) = result & 0xF8;
    }
    v4 = result & 7;
    if ( v4 == 3 )
    {
      v5 = 0;
      v6 = *((_BYTE *)v3 + 4);
      v7 = *((_BYTE *)v3 + 5);
      v8 = *((_BYTE *)v3 + 6);
      v9 = *((_BYTE *)v3 + 7);
      if ( *((_BYTE *)v3 + 9) )
        v5 = *(_QWORD *)&v3[2 * v9 + 4 + 2 * v8 + 2 * v7 + 2 * v6 + 2 * (__int64)*v3];
      v10 = 0;
      if ( *((_BYTE *)v3 + 8) )
        v10 = v3[2 * *v3 + 4 + 2 * v7 + 2 * v6 + 2 * (__int64)(v9 + v8)];
      if ( v9 )
        v11 = *(_QWORD *)&v3[2 * v8 + 4 + 2 * *v3 + 2 * (__int64)(v7 + v6)];
      else
        v11 = 0;
      if ( v8 )
        v12 = *(_QWORD *)&v3[2 * *v3 + 4 + 2 * (__int64)(v7 + v6)];
      else
        v12 = 0;
      if ( v7 )
        v13 = *(_QWORD *)&v3[2 * v6 + 4 + 2 * (__int64)*v3];
      else
        v13 = 0;
      if ( v6 )
        v3 = *(int **)&v3[2 * *v3 + 4];
      else
        v3 = 0;
    }
    else if ( v4 == 2 )
    {
      v13 = (__int64)v3;
      v5 = 0;
      v10 = 0;
      v11 = 0;
      v12 = 0;
      v3 = 0;
    }
    else
    {
      v5 = 0;
      v10 = 0;
      v11 = 0;
      v12 = 0;
      v13 = 0;
      if ( v4 != 1 )
        v3 = 0;
    }
    return sub_2E867B0(a1, a2, 0, 0, (__int64)v3, v13, v12, v11, v10, v5);
  }
  return result;
}
