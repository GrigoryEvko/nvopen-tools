// Function: sub_6DFF60
// Address: 0x6dff60
//
__int64 __fastcall sub_6DFF60(__int64 a1, int a2, _DWORD *a3)
{
  __int64 v4; // r12
  __int64 v5; // r13
  int v6; // ebx
  int v7; // ecx
  int v9; // eax
  __int64 v10; // r14
  int v11; // [rsp+0h] [rbp-40h]
  unsigned int v12; // [rsp+4h] [rbp-3Ch]
  int v13; // [rsp+Ch] [rbp-34h]

  v13 = a3[20];
  v12 = a3[26];
  if ( a1 )
  {
    v4 = a1;
    v5 = 0;
    v6 = 0;
    do
    {
      a3[20] = 0;
      sub_76CDC0(v4);
      v7 = a3[20];
      v6 |= v7;
      if ( v7 && !a3[26] )
      {
        if ( v5 )
        {
          if ( !a2 )
          {
            v9 = a3[27];
            v10 = v5;
            *((_QWORD *)a3 + 13) = 1;
            v11 = v9;
            while ( 1 )
            {
              sub_76CDC0(v10);
              if ( v10 == v4 )
                break;
              v10 = *(_QWORD *)(v10 + 16);
            }
            a3[27] = v11;
          }
        }
        else
        {
          v5 = v4;
        }
      }
      v4 = *(_QWORD *)(v4 + 16);
    }
    while ( v4 );
    v13 |= v6;
  }
  a3[20] = v13;
  a3[26] = v12;
  return v12;
}
