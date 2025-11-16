// Function: sub_38A2460
// Address: 0x38a2460
//
__int64 __fastcall sub_38A2460(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r12
  int v7; // eax
  char v8; // al
  unsigned __int64 v9; // r13
  unsigned int v10; // r14d
  __int64 *v11; // rdi
  __int64 v12; // rax
  int v14; // eax
  unsigned __int64 v15; // rsi
  int v16; // eax
  __int64 v17; // rcx
  int v18; // r9d
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rsi
  char v21; // [rsp+10h] [rbp-130h]
  char v22; // [rsp+18h] [rbp-128h]
  __int64 v25; // [rsp+30h] [rbp-110h] BYREF
  __int16 v26; // [rsp+38h] [rbp-108h]
  _QWORD v27[2]; // [rsp+40h] [rbp-100h] BYREF
  __int64 v28; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v29; // [rsp+58h] [rbp-E8h]
  __int64 v30; // [rsp+60h] [rbp-E0h]
  char *v31; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v32; // [rsp+78h] [rbp-C8h]
  _WORD v33[16]; // [rsp+80h] [rbp-C0h] BYREF
  char *v34; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v35; // [rsp+A8h] [rbp-98h]
  _WORD v36[16]; // [rsp+B0h] [rbp-90h] BYREF
  _QWORD *v37; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v38; // [rsp+D8h] [rbp-68h]
  _BYTE v39[32]; // [rsp+E0h] [rbp-60h] BYREF
  char v40; // [rsp+100h] [rbp-40h]

  v6 = a1 + 8;
  v28 = 0;
  v29 = 0;
  v30 = 0xFFFF;
  v25 = 0;
  v26 = 256;
  v37 = v39;
  v38 = 0x400000000LL;
  v40 = 0;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    goto LABEL_19;
  v7 = *(_DWORD *)(a1 + 64);
  if ( v7 != 13 )
  {
    if ( v7 == 372 )
    {
      do
      {
        if ( sub_2241AC0(a1 + 72, "tag") )
        {
          if ( sub_2241AC0(a1 + 72, "header") )
          {
            if ( sub_2241AC0(a1 + 72, "operands") )
            {
              v19 = *(_QWORD *)(a1 + 56);
              v31 = "invalid field '";
              v34 = (char *)&v31;
              v32 = a1 + 72;
              v33[0] = 1027;
              v35 = (__int64)"'";
              v36[0] = 770;
              v8 = sub_38814C0(v6, v19, (__int64)&v34);
            }
            else
            {
              v27[1] = 8;
              v27[0] = "operands";
              if ( v40 )
              {
                v20 = *(_QWORD *)(a1 + 56);
                v34 = "field '";
                v35 = (__int64)v27;
                v31 = (char *)&v34;
                v32 = (__int64)"' cannot be specified more than once";
                v36[0] = 1283;
                v33[0] = 770;
                v8 = sub_38814C0(v6, v20, (__int64)&v31);
              }
              else
              {
                v16 = sub_3887100(v6);
                v32 = 0x400000000LL;
                *(_DWORD *)(a1 + 64) = v16;
                v31 = (char *)v33;
                v8 = sub_38A2250(a1, (__int64)&v31, a4, a5, a6);
                if ( !v8 )
                {
                  v35 = 0x400000000LL;
                  v34 = (char *)v36;
                  if ( (_DWORD)v32 )
                  {
                    sub_38874C0((__int64)&v34, &v31, (__int64)v36, v17, (int)&v34, v18);
                    v8 = 0;
                  }
                  v21 = v8;
                  v40 = 1;
                  sub_38874C0((__int64)&v37, &v34, (__int64)v36, v17, (int)&v34, v18);
                  v8 = v21;
                  if ( v34 != (char *)v36 )
                  {
                    _libc_free((unsigned __int64)v34);
                    v8 = v21;
                  }
                }
                if ( v31 != (char *)v33 )
                {
                  v22 = v8;
                  _libc_free((unsigned __int64)v31);
                  v8 = v22;
                }
              }
            }
          }
          else
          {
            v8 = sub_388B8F0(a1, (__int64)"header", 6, (__int64)&v25);
          }
        }
        else
        {
          v8 = sub_38899B0(a1, (__int64)"tag", 3, (__int64)&v28);
        }
        if ( v8 )
          goto LABEL_19;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v14 = sub_3887100(v6);
        *(_DWORD *)(a1 + 64) = v14;
      }
      while ( v14 == 372 );
    }
    v15 = *(_QWORD *)(a1 + 56);
    v34 = "expected field label here";
    v36[0] = 259;
    if ( (unsigned __int8)sub_38814C0(v6, v15, (__int64)&v34) )
    {
LABEL_19:
      v10 = 1;
      goto LABEL_13;
    }
  }
LABEL_8:
  v9 = *(_QWORD *)(a1 + 56);
  v10 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v10 )
  {
    if ( (_BYTE)v29 )
    {
      v11 = *(__int64 **)a1;
      if ( a3 )
        v12 = sub_15BA790(v11, v28, v25, v37, (unsigned int)v38, 1u, 1);
      else
        v12 = sub_15BA790(v11, v28, v25, v37, (unsigned int)v38, 0, 1);
      *a2 = v12;
    }
    else
    {
      v34 = "missing required field 'tag'";
      v36[0] = 259;
      v10 = sub_38814C0(v6, v9, (__int64)&v34);
    }
  }
LABEL_13:
  if ( v37 != (_QWORD *)v39 )
    _libc_free((unsigned __int64)v37);
  return v10;
}
