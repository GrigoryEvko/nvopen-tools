// Function: sub_5CB280
// Address: 0x5cb280
//
__int64 __fastcall sub_5CB280(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // r13
  int v7; // ebx
  __int64 v8; // rax
  __int64 *v9; // r15
  __int64 v10; // r13
  unsigned __int64 v11; // r13
  __int64 **v12; // r15
  char v13; // al
  __int64 *v14; // rbx
  int v15; // r14d
  int v16; // r8d
  int v17; // r9d
  int v19; // eax
  __int64 v20; // rax
  int v21; // eax
  int v22; // eax
  __int64 v23; // rdx
  _QWORD *v24; // rax
  int v25; // [rsp+8h] [rbp-68h]
  int v26; // [rsp+8h] [rbp-68h]
  int v27; // [rsp+Ch] [rbp-64h]
  int v28; // [rsp+Ch] [rbp-64h]
  char v30; // [rsp+1Fh] [rbp-51h]
  int v31; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+28h] [rbp-48h] BYREF
  __int64 v33[7]; // [rsp+38h] [rbp-38h] BYREF

  v32 = a2;
  v3 = sub_5C7B50(a1, (__int64)&v32, a3);
  v4 = *(_QWORD *)(a1 + 32);
  v5 = v3;
  v6 = *(_QWORD *)(v4 + 40);
  if ( (unsigned int)sub_7ADED0("printf") )
  {
    v8 = 0;
  }
  else if ( (unsigned int)sub_7ADED0("scanf") )
  {
    v8 = 1;
  }
  else
  {
    v7 = sub_7ADED0("strftime");
    v8 = 2;
    if ( !v7 )
    {
      sub_684B10(1217, v4 + 24, v6);
      *(_BYTE *)(a1 + 8) = 0;
      v30 = 0;
      goto LABEL_5;
    }
  }
  LOBYTE(v7) = 1;
  v30 = *((_BYTE *)&off_496EE00 + 16 * v8 + 8);
LABEL_5:
  v9 = *(__int64 **)v4;
  if ( !(unsigned int)sub_5CACA0((__int64)v9, a1, 0, 0x7FFFFFFF, v33) )
  {
    sub_5CACA0(*v9, a1, 0, 0x7FFFFFFF, v33);
    return v32;
  }
  v10 = LODWORD(v33[0]);
  if ( (unsigned int)sub_5CACA0(*v9, a1, 0, 0x7FFFFFFF, v33) )
  {
    v11 = ((unsigned __int64)LODWORD(v33[0]) << 32) | v10;
    if ( v5 )
    {
      if ( (v7 & 1) != 0 )
      {
        v12 = *(__int64 ***)(v5 + 168);
        v13 = *((_BYTE *)v12 + 16);
        if ( (v13 & 2) != 0 )
        {
          v14 = *v12;
          v15 = v12[5] != 0;
          v16 = v11;
          v17 = HIDWORD(v11);
          if ( !*v12 )
            goto LABEL_38;
          v31 = 0;
          while ( 1 )
          {
            if ( ++v15 == v16 )
            {
              v25 = v17;
              v27 = v16;
              v19 = sub_8D2E30(v14[1]);
              v16 = v27;
              v17 = v25;
              if ( !v19 || (v20 = sub_8D46C0(v14[1]), v21 = sub_8D29E0(v20), v16 = v27, v17 = v25, !v21) )
              {
                v26 = v17;
                v28 = v16;
                sub_6851C0(1138, **(_QWORD **)(a1 + 32) + 24LL);
                *(_BYTE *)(a1 + 8) = 0;
                v16 = v28;
                v17 = v26;
              }
            }
            if ( v15 != v17 )
              goto LABEL_12;
            if ( v14[10] )
            {
              v14 = (__int64 *)*v14;
              v31 = 1;
              if ( !v14 )
              {
LABEL_17:
                if ( v31 )
                {
                  if ( v15 >= (int)v11 )
                    goto LABEL_19;
LABEL_43:
                  sub_6851C0(1137, **(_QWORD **)(a1 + 32) + 24LL);
                  *(_BYTE *)(a1 + 8) = 0;
LABEL_19:
                  if ( !HIDWORD(v11) )
                    return v32;
                  if ( ((_BYTE)v12[2] & 1) != 0 )
                  {
                    if ( SHIDWORD(v11) <= 0 )
                      return v32;
                    if ( (v31 & 1) == 0 )
                      goto LABEL_33;
                  }
                  else
                  {
LABEL_29:
                    if ( !v31 )
                    {
                      v23 = a1 + 56;
                      if ( unk_4F077B4 )
                      {
                        sub_684AA0(5, 1135, v23);
                      }
                      else
                      {
                        sub_684AA0(8, 1135, v23);
                        *(_BYTE *)(a1 + 8) = 0;
                      }
                      if ( SHIDWORD(v11) <= 0 )
                        return v32;
LABEL_33:
                      if ( v15 + 1 != HIDWORD(v11) && (!unk_4F077B4 || (int)v11 >= SHIDWORD(v11)) )
                      {
                        sub_6851C0(1136, ***(_QWORD ***)(a1 + 32) + 24LL);
                        *(_BYTE *)(a1 + 8) = 0;
                        return v32;
                      }
LABEL_47:
                      if ( *(_BYTE *)(a1 + 8) )
                      {
                        *((_DWORD *)v12 + 7) = v11;
                        *((_BYTE *)v12 + 24) = v30;
                        *((_DWORD *)v12 + 8) = HIDWORD(v11);
                      }
                      return v32;
                    }
                  }
                  if ( SHIDWORD(v11) <= 0 )
                    return v32;
                  goto LABEL_47;
                }
                v13 = *((_BYTE *)v12 + 16);
LABEL_38:
                if ( (v13 & 1) != 0 || v15 + 1 != HIDWORD(v11) || a3 != 11 )
                {
                  v31 = 0;
                  if ( v15 < (int)v11 )
                    goto LABEL_43;
                  goto LABEL_19;
                }
                v31 = 0;
                if ( (*(_BYTE *)(v32 + 195) & 9) == 1 )
                {
                  v24 = **(_QWORD ***)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v32 + 248) + 192LL) + 152LL) + 168LL);
                  if ( v24 )
                  {
                    do
                    {
                      if ( v24[10] )
                      {
                        v31 = 1;
                        goto LABEL_42;
                      }
                      v24 = (_QWORD *)*v24;
                    }
                    while ( v24 );
                    v31 = 0;
                  }
                }
LABEL_42:
                if ( v15 < (int)v11 )
                  goto LABEL_43;
                goto LABEL_29;
              }
            }
            else
            {
              v22 = 1;
              if ( (*((_BYTE *)v14 + 33) & 2) == 0 )
                v22 = v31;
              v31 = v22;
LABEL_12:
              v14 = (__int64 *)*v14;
              if ( !v14 )
                goto LABEL_17;
            }
          }
        }
      }
    }
  }
  return v32;
}
