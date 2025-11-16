// Function: sub_2C893B0
// Address: 0x2c893b0
//
__int64 __fastcall sub_2C893B0(__int64 a1)
{
  unsigned int v1; // r12d
  _QWORD *v2; // rax
  _BYTE *v3; // rdx
  __int64 *v4; // rdx
  __int64 v5; // r15
  __int64 v6; // r8
  unsigned int v7; // ebx
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  int v13; // edx
  int v14; // eax
  _QWORD *v15; // rdi
  unsigned __int8 v17; // cl
  unsigned int i; // eax
  unsigned int v19; // eax
  __int64 v20; // [rsp+8h] [rbp-68h]
  unsigned int v21; // [rsp+14h] [rbp-5Ch]
  _BYTE **v22; // [rsp+20h] [rbp-50h] BYREF
  __int64 v23; // [rsp+28h] [rbp-48h]
  _QWORD v24[8]; // [rsp+30h] [rbp-40h] BYREF

  v1 = 0;
  v23 = 0x200000000LL;
  v2 = *(_QWORD **)a1;
  v22 = (_BYTE **)v24;
  v3 = (_BYTE *)*v2;
  if ( *(_BYTE *)*v2 > 0x1Cu )
  {
    v4 = (v3[7] & 0x40) != 0 ? (__int64 *)*((_QWORD *)v3 - 1) : (__int64 *)&v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
    v5 = *v4;
    v6 = *(unsigned int *)(a1 + 8);
    v1 = 1;
    LODWORD(v23) = 1;
    v24[0] = v5;
    if ( (_DWORD)v6 != 1 )
    {
      v7 = 1;
      v8 = v2[1];
      if ( *(_BYTE *)v8 <= 0x1Cu )
      {
LABEL_14:
        v15 = v22;
        v1 = 0;
      }
      else
      {
        while ( 1 )
        {
          if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
            v9 = *(__int64 **)(v8 - 8);
          else
            v9 = (__int64 *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
          v10 = *v9;
          v11 = (unsigned int)v23;
          v12 = (unsigned int)v23 + 1LL;
          if ( v12 > HIDWORD(v23) )
          {
            v20 = v10;
            v21 = v6;
            sub_C8D5F0((__int64)&v22, v24, v12, 8u, v6, v10);
            v11 = (unsigned int)v23;
            v10 = v20;
            v6 = v21;
          }
          if ( v5 != v10 )
            v1 = 0;
          ++v7;
          v22[v11] = (_BYTE *)v10;
          v13 = v23;
          v14 = v23 + 1;
          LODWORD(v23) = v23 + 1;
          if ( (_DWORD)v6 == v7 )
            break;
          v8 = *(_QWORD *)(*(_QWORD *)a1 + 8LL * v7);
          if ( *(_BYTE *)v8 <= 0x1Cu )
            goto LABEL_14;
        }
        v15 = v22;
        if ( (_BYTE)v1 )
          goto LABEL_15;
        v17 = **v22;
        if ( v17 <= 0x1Cu || v17 != 63 && (unsigned __int8)(v17 - 60) > 1u )
          goto LABEL_15;
        if ( v14 != 1 )
        {
          for ( i = 1; ; ++i )
          {
            if ( v17 == 60 )
            {
              if ( *v22[i] != 60 )
                goto LABEL_27;
            }
            else if ( v17 == 61 )
            {
              if ( *v22[i] != 61 )
                goto LABEL_27;
            }
            else if ( v17 != 63 || *v22[i] != 63 )
            {
LABEL_27:
              v1 = 0;
              goto LABEL_15;
            }
            if ( v13 == i )
              break;
          }
        }
        v1 = 1;
        if ( v17 != 60 )
        {
          v19 = sub_2C893B0(&v22);
          v15 = v22;
          v1 = v19;
        }
      }
LABEL_15:
      if ( v15 != v24 )
        _libc_free((unsigned __int64)v15);
    }
  }
  return v1;
}
