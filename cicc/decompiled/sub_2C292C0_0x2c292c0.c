// Function: sub_2C292C0
// Address: 0x2c292c0
//
void __fastcall sub_2C292C0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r13
  __int64 *v3; // r12
  __int64 v4; // r15
  _QWORD *v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rbx
  _QWORD *v9; // rbx
  _BYTE *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 i; // rcx
  __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // [rsp+10h] [rbp-A0h]
  _QWORD *v18; // [rsp+18h] [rbp-98h]
  _QWORD *v19; // [rsp+18h] [rbp-98h]
  __int64 v20; // [rsp+28h] [rbp-88h] BYREF
  __int64 v21; // [rsp+30h] [rbp-80h] BYREF
  int v22; // [rsp+38h] [rbp-78h]
  _QWORD v23[2]; // [rsp+40h] [rbp-70h] BYREF
  void *v24[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v25; // [rsp+70h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 88) != 1 || (v16 = *(_QWORD *)(a1 + 80), *(_BYTE *)(v16 + 4)) || *(_DWORD *)v16 != 1 )
  {
    v1 = sub_2BF3F10((_QWORD *)a1);
    v2 = v1;
    if ( v1 )
    {
      if ( *(_DWORD *)(v1 + 64) == 1 )
        v2 = **(_QWORD **)(v1 + 56);
      else
        v2 = 0;
    }
    v3 = *(__int64 **)(a1 + 416);
    v17 = &v3[*(unsigned int *)(a1 + 424)];
    while ( v17 != v3 )
    {
      v4 = *v3;
      v5 = *(_QWORD **)(*v3 + 16);
      v6 = 8LL * *(unsigned int *)(*v3 + 24);
      v18 = &v5[(unsigned __int64)v6 / 8];
      v7 = v6 >> 3;
      v8 = v6 >> 5;
      if ( v8 )
      {
        v9 = &v5[4 * v8];
        while ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v5 + 16LL))(*v5, v4) )
        {
          if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v5[1] + 16LL))(v5[1], v4) )
          {
            ++v5;
            break;
          }
          if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v5[2] + 16LL))(v5[2], v4) )
          {
            v5 += 2;
            break;
          }
          if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v5[3] + 16LL))(v5[3], v4) )
          {
            v5 += 3;
            break;
          }
          v5 += 4;
          if ( v9 == v5 )
          {
            v7 = v18 - v5;
            goto LABEL_27;
          }
        }
LABEL_13:
        if ( v18 != v5 )
        {
          v10 = *(_BYTE **)(v4 + 40);
          if ( v10 )
          {
            if ( *v10 > 0x15u )
            {
              v11 = *(_QWORD *)(v4 + 16);
              v12 = v2 + 112;
              for ( i = v11 + 8LL * *(unsigned int *)(v4 + 24); i != v11; v11 += 8 )
              {
                if ( !*(_QWORD *)v11 )
                  BUG();
                if ( *(_QWORD *)(*(_QWORD *)v11 + 40LL) == v2 )
                  v12 = *(_QWORD *)(v2 + 120);
              }
              v23[1] = v12;
              v23[0] = v2;
              v25 = 257;
              v21 = 0;
              v20 = v4;
              v19 = sub_2C27AE0(v23, 80, &v20, 1, v22, 0, &v21, v24);
              sub_9C6650(&v21);
              v14 = (__int64)v19;
              v24[0] = (void *)v4;
              v24[1] = v19;
              if ( v19 )
                v14 = (__int64)(v19 + 12);
              sub_2BF1090(v4, v14, (unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))sub_2C255A0, (__int64)v24);
            }
          }
        }
        goto LABEL_16;
      }
LABEL_27:
      if ( v7 != 2 )
      {
        if ( v7 != 3 )
        {
          if ( v7 == 1 && !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v5 + 16LL))(*v5, v4) )
            goto LABEL_13;
          goto LABEL_16;
        }
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v5 + 16LL))(*v5, v4) )
          goto LABEL_13;
        ++v5;
      }
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v5 + 16LL))(*v5, v4) )
        goto LABEL_13;
      v15 = v5[1];
      ++v5;
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 16LL))(v15, v4) )
        goto LABEL_13;
LABEL_16:
      ++v3;
    }
  }
}
