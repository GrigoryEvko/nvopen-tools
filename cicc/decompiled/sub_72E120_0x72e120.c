// Function: sub_72E120
// Address: 0x72e120
//
__int64 __fastcall sub_72E120(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rbx
  int v7; // r12d
  unsigned int v8; // r14d
  __int64 v9; // rdi
  unsigned __int8 v10; // al
  int v11; // r13d
  __int64 v12; // rdi
  int v13; // eax
  __int64 v15; // rax
  int v16; // eax

  if ( a1 )
  {
    v6 = a1;
    v7 = 1;
    v8 = 0;
    while ( 1 )
    {
      v10 = *((_BYTE *)v6 + 8);
      v11 = v7 + 1;
      if ( v10 == 2 )
      {
        ++v7;
        if ( v6[4] )
        {
          v15 = sub_89A800();
          v8 += v11 * sub_72E220(v15);
        }
        goto LABEL_5;
      }
      if ( v10 > 2u )
      {
        if ( v10 != 3 )
          sub_721090();
        goto LABEL_5;
      }
      if ( v10 )
        break;
      v12 = v6[4];
      ++v7;
      if ( v12 )
      {
        v13 = sub_72E3F0(v12);
        v6 = (__int64 *)*v6;
        v8 += v11 * v13;
        if ( !v6 )
          return v8;
      }
      else
      {
LABEL_5:
        v6 = (__int64 *)*v6;
        if ( !v6 )
          return v8;
      }
    }
    v9 = v6[4];
    if ( (v6[3] & 1) != 0 )
    {
      a4 = (unsigned int)(3 * v7++);
      v8 += ((_DWORD)v9 + 1) << a4;
    }
    else if ( v9 )
    {
      v16 = sub_72DB90(v9, a2, a3, a4, a5, a6);
      a4 = (unsigned int)(3 * v7++);
      v8 += (v16 + 1) << a4;
    }
    else
    {
      ++v7;
    }
    goto LABEL_5;
  }
  return 0;
}
