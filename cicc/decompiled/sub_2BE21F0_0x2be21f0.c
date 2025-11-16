// Function: sub_2BE21F0
// Address: 0x2be21f0
//
void __fastcall sub_2BE21F0(_QWORD *a1, unsigned __int8 a2, __int64 a3)
{
  __int64 v3; // rax
  char *v4; // r15
  char *v5; // rbx
  __int64 v6; // r13
  char *v7; // r8
  char *v8; // r9
  char *v9; // rax
  __int64 v10; // rax
  size_t v11; // rdx
  __int64 v12; // rsi
  volatile signed __int32 **v13; // rsi
  __int64 v14; // r14
  char *v15; // r8
  char *v16; // r9
  char *v17; // rdx
  unsigned int v18; // r15d
  __int64 v19; // [rsp+0h] [rbp-70h]
  char *v20; // [rsp+8h] [rbp-68h]
  char *v21; // [rsp+10h] [rbp-60h]
  char *v22; // [rsp+10h] [rbp-60h]
  char *v23; // [rsp+18h] [rbp-58h]
  char *v24; // [rsp+18h] [rbp-58h]
  char v25; // [rsp+20h] [rbp-50h]
  bool v26; // [rsp+2Bh] [rbp-45h]
  volatile signed __int32 *v28[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = *a1 + 24LL * *(_QWORD *)(*(_QWORD *)(a1[7] + 56LL) + 48 * a3 + 16);
  if ( *(_BYTE *)(v3 + 16) )
  {
    v4 = (char *)a1[3];
    v5 = (char *)a1[5];
    v6 = *(_QWORD *)(a1[7] + 56LL) + 48 * a3;
    v7 = *(char **)v3;
    v8 = *(char **)(v3 + 8);
    if ( v4 != v5 )
    {
      v9 = (char *)a1[3];
      while ( v9 != (char *)(&v4[(_QWORD)v8] - v7) )
      {
        if ( v5 == ++v9 )
          goto LABEL_7;
      }
      v5 = v9;
    }
LABEL_7:
    v10 = a1[6];
    v11 = v5 - v4;
    v12 = *(_QWORD *)(v10 + 16);
    LODWORD(v10) = *(_DWORD *)v10 & 1;
    v26 = v10;
    if ( (_DWORD)v10 )
    {
      v13 = (volatile signed __int32 **)(v12 + 80);
      v19 = v8 - v7;
      v21 = v8;
      v23 = v7;
      sub_2208E20(v28, v13);
      v14 = sub_222F790(v28, (__int64)v13);
      sub_2209150(v28);
      v15 = v23;
      v16 = v21;
      if ( v5 - v4 != v19 )
        return;
      if ( v23 == v21 )
        goto LABEL_12;
      v17 = v4;
      while ( 1 )
      {
        v18 = *v17;
        v20 = v16;
        v22 = v17;
        v24 = v15;
        v25 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v14 + 32LL))(v14, (unsigned int)*v15);
        if ( v25 != (*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v14 + 32LL))(v14, v18) )
          break;
        v16 = v20;
        v15 = v24 + 1;
        v17 = v22 + 1;
        if ( v20 == v24 + 1 )
          goto LABEL_11;
      }
    }
    else
    {
      if ( v11 != v8 - v7 )
        return;
      if ( !v11 )
        goto LABEL_13;
      v26 = memcmp(v7, v4, v11) == 0;
LABEL_11:
      if ( v26 )
      {
LABEL_12:
        v4 = (char *)a1[3];
LABEL_13:
        if ( v4 == v5 )
        {
          sub_2BE1DD0((__int64)a1, a2, *(_QWORD *)(v6 + 8));
        }
        else
        {
          a1[3] = v5;
          sub_2BE1DD0((__int64)a1, a2, *(_QWORD *)(v6 + 8));
          a1[3] = v4;
        }
      }
    }
  }
}
