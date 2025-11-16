// Function: sub_13FC570
// Address: 0x13fc570
//
__int64 __fastcall sub_13FC570(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  unsigned int v7; // eax
  unsigned int v8; // r15d
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // r9
  __int64 v14; // rdi
  unsigned __int8 v15; // [rsp+Fh] [rbp-41h]
  _QWORD *v16; // [rsp+10h] [rbp-40h]
  _QWORD *v17; // [rsp+18h] [rbp-38h]

  LOBYTE(v7) = sub_13FC1A0(a1, a2);
  v8 = v7;
  if ( !(_BYTE)v7 )
  {
    v15 = sub_14AF470(a2, 0, 0, 0);
    if ( v15 )
    {
      if ( !(unsigned __int8)sub_15F2ED0(a2) )
      {
        v10 = (unsigned int)*(unsigned __int8 *)(a2 + 16) - 34;
        if ( (unsigned int)v10 > 0x36 || (v11 = 0x40018000000001LL, !_bittest64(&v11, v10)) )
        {
          if ( !a4 )
          {
            v14 = sub_13FC520(a1);
            if ( !v14 )
              return v8;
            a4 = sub_157EBA0(v14);
          }
          v12 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          {
            v13 = *(_QWORD **)(a2 - 8);
            v16 = &v13[v12];
          }
          else
          {
            v16 = (_QWORD *)a2;
            v13 = (_QWORD *)(a2 - v12 * 8);
          }
          if ( v13 == v16 )
          {
LABEL_13:
            sub_15F22F0(a2, a4);
            sub_1624960(a2, 0, 0);
            *a3 = 1;
            return v15;
          }
          else
          {
            while ( 1 )
            {
              v17 = v13;
              v8 = sub_13FC6C0(a1, *v13, a3, a4);
              if ( !(_BYTE)v8 )
                break;
              v13 = v17 + 3;
              if ( v16 == v17 + 3 )
                goto LABEL_13;
            }
          }
        }
      }
    }
  }
  return v8;
}
