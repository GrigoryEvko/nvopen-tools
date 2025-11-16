// Function: sub_358F5F0
// Address: 0x358f5f0
//
__int64 __fastcall sub_358F5F0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rax
  int v12; // ecx
  __int64 v13; // rsi
  int v14; // ecx
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  unsigned int v19; // ebx
  unsigned int v20; // eax
  __int64 result; // rax
  unsigned int v22; // edx
  int v23; // eax
  int v24; // r8d
  int v25; // r11d
  unsigned int v26; // edx
  _QWORD *v27; // rax
  __int64 v28; // rdi
  __int64 *v29; // rax
  int v30; // edi
  __int64 v31; // rax
  unsigned int v32; // edx
  __int64 v33; // rdi
  int v34; // r11d
  __int64 v36; // [rsp+10h] [rbp-90h]
  __int64 v37; // [rsp+10h] [rbp-90h]
  unsigned __int64 v38; // [rsp+18h] [rbp-88h]
  __int64 v39; // [rsp+18h] [rbp-88h]
  __int64 v40; // [rsp+20h] [rbp-80h] BYREF
  _QWORD *v41; // [rsp+28h] [rbp-78h] BYREF
  __int64 v42[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v43; // [rsp+40h] [rbp-60h]
  __int64 v44; // [rsp+50h] [rbp-50h] BYREF
  __int64 v45; // [rsp+58h] [rbp-48h]
  __int64 v46; // [rsp+60h] [rbp-40h]
  unsigned int v47; // [rsp+68h] [rbp-38h]

  v6 = a2 + 320;
  v7 = a2;
  v9 = *(_QWORD *)(a2 + 328);
  if ( LOBYTE(qword_500BA28[8]) )
  {
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v47 = 0;
    if ( v9 == v6 )
    {
LABEL_34:
      v42[0] = a2;
      v42[1] = a1 + 1056;
      v43 = &v44;
      sub_358C950(v42, a1 + 40, a1 + 72, (__int64)a4, a5, a6);
      return sub_C7D6A0(v45, 16LL * v47, 8);
    }
    while ( 1 )
    {
      while ( 1 )
      {
        sub_3586670((__int64)v42, (void (__fastcall ***)(unsigned __int64 *, _QWORD, __int64))a1, v9);
        if ( ((unsigned __int8)v43 & 1) == 0 )
          break;
        v9 = *(_QWORD *)(v9 + 8);
        if ( v9 == v6 )
          goto LABEL_34;
      }
      a5 = v42[0];
      v40 = v9;
      if ( !v47 )
        break;
      a6 = v45;
      v25 = 1;
      a4 = 0;
      v26 = (v47 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v27 = (_QWORD *)(v45 + 16LL * v26);
      v28 = *v27;
      if ( v9 != *v27 )
      {
        while ( v28 != -4096 )
        {
          if ( v28 == -8192 && !a4 )
            a4 = v27;
          v26 = (v47 - 1) & (v25 + v26);
          v27 = (_QWORD *)(v45 + 16LL * v26);
          v28 = *v27;
          if ( v9 == *v27 )
            goto LABEL_32;
          ++v25;
        }
        if ( !a4 )
          a4 = v27;
        ++v44;
        v30 = v46 + 1;
        v41 = a4;
        if ( 4 * ((int)v46 + 1) < 3 * v47 )
        {
          v31 = v9;
          a6 = v47 >> 3;
          if ( v47 - HIDWORD(v46) - v30 <= (unsigned int)a6 )
          {
            v37 = v42[0];
            sub_2E3E470((__int64)&v44, v47);
            sub_3585550((__int64)&v44, &v40, &v41);
            v31 = v40;
            a5 = v37;
            v30 = v46 + 1;
            a4 = v41;
          }
LABEL_46:
          LODWORD(v46) = v30;
          if ( *a4 != -4096 )
            --HIDWORD(v46);
          *a4 = v31;
          v29 = a4 + 1;
          a4[1] = 0;
          goto LABEL_33;
        }
LABEL_50:
        v36 = v42[0];
        sub_2E3E470((__int64)&v44, 2 * v47);
        a5 = v36;
        if ( v47 )
        {
          v31 = v40;
          v32 = (v47 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          a4 = (_QWORD *)(v45 + 16LL * v32);
          v33 = *a4;
          if ( v40 == *a4 )
          {
LABEL_52:
            v41 = a4;
            v30 = v46 + 1;
          }
          else
          {
            v34 = 1;
            a6 = 0;
            while ( v33 != -4096 )
            {
              if ( !a6 && v33 == -8192 )
                a6 = (__int64)a4;
              v32 = (v47 - 1) & (v34 + v32);
              a4 = (_QWORD *)(v45 + 16LL * v32);
              v33 = *a4;
              if ( v40 == *a4 )
                goto LABEL_52;
              ++v34;
            }
            if ( !a6 )
              a6 = (__int64)a4;
            v41 = (_QWORD *)a6;
            v30 = v46 + 1;
            a4 = (_QWORD *)a6;
          }
        }
        else
        {
          v31 = v40;
          v41 = 0;
          v30 = v46 + 1;
          a4 = 0;
        }
        goto LABEL_46;
      }
LABEL_32:
      v29 = v27 + 1;
LABEL_33:
      *v29 = a5;
      v9 = *(_QWORD *)(v9 + 8);
      if ( v9 == v6 )
        goto LABEL_34;
    }
    ++v44;
    v41 = 0;
    goto LABEL_50;
  }
  v10 = a1 + 40;
  if ( v9 == v6 )
    goto LABEL_12;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(a1 + 1016);
      v42[0] = v9;
      v12 = *(_DWORD *)(v11 + 24);
      v13 = *(_QWORD *)(v11 + 8);
      if ( v12 )
      {
        v14 = v12 - 1;
        v15 = v14 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v16 = (__int64 *)(v13 + 16LL * v15);
        v17 = *v16;
        if ( v9 != *v16 )
        {
          v23 = 1;
          while ( v17 != -4096 )
          {
            v24 = v23 + 1;
            v15 = v14 & (v23 + v15);
            v16 = (__int64 *)(v13 + 16LL * v15);
            v17 = *v16;
            if ( v9 == *v16 )
              goto LABEL_7;
            v23 = v24;
          }
          goto LABEL_4;
        }
LABEL_7:
        v18 = v16[1];
        if ( v18 )
        {
          v44 = **(_QWORD **)(v18 + 32);
          if ( v44 )
          {
            v38 = *sub_3588500(v10, v42);
            if ( v38 > *sub_3588500(v10, &v44) )
              break;
          }
        }
      }
LABEL_4:
      v9 = *(_QWORD *)(v9 + 8);
      if ( v9 == v6 )
        goto LABEL_11;
    }
    v39 = *sub_3588500(v10, v42);
    *sub_3588500(v10, &v44) = v39;
    v9 = *(_QWORD *)(v9 + 8);
  }
  while ( v9 != v6 );
LABEL_11:
  v7 = a2;
LABEL_12:
  v19 = 0;
  do
    v20 = v19++;
  while ( LODWORD(qword_500BDA8[8]) > v20 && (unsigned __int8)sub_358DCD0(a1, v7, 0) );
  *(_DWORD *)(a1 + 400) = 0;
  sub_3583FB0(*(_QWORD *)(a1 + 936));
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 944) = a1 + 928;
  *(_QWORD *)(a1 + 952) = a1 + 928;
  *(_QWORD *)(a1 + 960) = 0;
  do
  {
    result = LODWORD(qword_500BDA8[8]);
    v22 = v19++;
    if ( LODWORD(qword_500BDA8[8]) <= v22 )
      goto LABEL_19;
  }
  while ( (unsigned __int8)sub_358DCD0(a1, v7, 0) );
  do
  {
    result = LODWORD(qword_500BDA8[8]);
LABEL_19:
    if ( v19 >= (unsigned int)result )
      break;
    ++v19;
    result = sub_358DCD0(a1, v7, 1);
  }
  while ( (_BYTE)result );
  return result;
}
