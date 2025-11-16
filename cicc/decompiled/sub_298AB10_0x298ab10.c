// Function: sub_298AB10
// Address: 0x298ab10
//
__int64 __fastcall sub_298AB10(__int64 *a1, char a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r14
  __int64 *v4; // rbx
  __int64 v5; // rsi
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rcx
  _QWORD *v12; // rsi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 *v20; // rcx
  __int64 v21; // r12
  __int64 *v22; // rbx
  __int64 *v23; // rax
  __int64 *v24; // r14
  char v25; // r9
  __int64 *v26; // rcx
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // esi
  __int64 v31; // rdx
  __int64 v32; // r8
  unsigned int v33; // edi
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v37; // [rsp+10h] [rbp-D0h]
  __int64 v38; // [rsp+18h] [rbp-C8h]
  __int64 v40; // [rsp+28h] [rbp-B8h]
  __int64 *v41; // [rsp+30h] [rbp-B0h]
  __int64 v42; // [rsp+38h] [rbp-A8h]
  char v44; // [rsp+47h] [rbp-99h]
  __int64 *v45; // [rsp+48h] [rbp-98h]
  __int64 v46; // [rsp+58h] [rbp-88h] BYREF
  __int64 v47; // [rsp+60h] [rbp-80h] BYREF
  __int64 v48; // [rsp+68h] [rbp-78h] BYREF
  _BYTE v49[112]; // [rsp+70h] [rbp-70h] BYREF

  v38 = a1[1];
  if ( a2 )
  {
    v2 = a1 + 100;
  }
  else
  {
    v38 = a1[2];
    v2 = a1 + 82;
  }
  v3 = (__int64 *)v49;
  sub_11D2BF0((__int64)v49, 0);
  v41 = (__int64 *)*v2;
  v37 = *v2 + 8LL * *((unsigned int *)v2 + 2);
  if ( *v2 != v37 )
  {
    do
    {
      v10 = *v41;
      v40 = *(_QWORD *)(*v41 + 40);
      v46 = *(_QWORD *)(*v41 - 32);
      v47 = *(_QWORD *)(v10 - 64);
      if ( a2 )
      {
        v4 = sub_298A890((__int64)(a1 + 96), &v47);
        if ( *((_DWORD *)v4 + 4) == 1 )
          goto LABEL_21;
      }
      else
      {
        v4 = sub_298A890((__int64)(a1 + 78), &v46);
        if ( *((_DWORD *)v4 + 4) == 1 )
        {
LABEL_21:
          v11 = v4[1];
          v12 = (_QWORD *)(v11 + 32LL * *((unsigned int *)v4 + 6));
          if ( (_QWORD *)v11 == v12 )
          {
            if ( v40 == *(_QWORD *)v11 )
              goto LABEL_31;
          }
          else
          {
            v13 = (_QWORD *)v4[1];
            while ( 1 )
            {
              v14 = *v13;
              v15 = v13;
              if ( *v13 != -4096 && v14 != -8192 )
                break;
              v13 += 4;
              if ( v12 == v13 )
              {
                v14 = v15[4];
                break;
              }
            }
            if ( v14 == v40 )
            {
              do
              {
                if ( *(_QWORD *)v11 != -8192 && *(_QWORD *)v11 != -4096 )
                  break;
                v11 += 32;
              }
              while ( v12 != (_QWORD *)v11 );
LABEL_31:
              v16 = *(_QWORD *)(v11 + 8);
              if ( *(_QWORD *)(v10 - 96) )
              {
                v17 = *(_QWORD *)(v10 - 88);
                **(_QWORD **)(v10 - 80) = v17;
                if ( v17 )
                  *(_QWORD *)(v17 + 16) = *(_QWORD *)(v10 - 80);
              }
              *(_QWORD *)(v10 - 96) = v16;
              if ( v16 )
              {
                v18 = *(_QWORD *)(v16 + 16);
                *(_QWORD *)(v10 - 88) = v18;
                if ( v18 )
                  *(_QWORD *)(v18 + 16) = v10 - 88;
                *(_QWORD *)(v10 - 80) = v16 + 16;
                *(_QWORD *)(v16 + 16) = v10 - 96;
              }
              if ( *(_BYTE *)(v11 + 24) )
              {
                v48 = *(_QWORD *)(v11 + 16);
                sub_BC8EC0(v10, (unsigned int *)&v48, 2, 0);
              }
              goto LABEL_18;
            }
          }
        }
      }
      sub_11D2C80(v3, *a1, (unsigned __int8 *)byte_3F871B3, 0);
      v5 = v40;
      if ( a2 )
        v5 = v47;
      sub_11D33F0(v3, v5, v38);
      if ( !*((_DWORD *)v4 + 4) )
        goto LABEL_9;
      v19 = (__int64 *)v4[1];
      v20 = &v19[4 * *((unsigned int *)v4 + 6)];
      if ( v19 == v20 )
        goto LABEL_9;
      while ( 1 )
      {
        v21 = *v19;
        v22 = v19;
        if ( *v19 != -8192 && v21 != -4096 )
          break;
        v19 += 4;
        if ( v20 == v19 )
          goto LABEL_9;
      }
      if ( v19 == v20 )
      {
LABEL_9:
        v6 = v40;
        goto LABEL_10;
      }
      v23 = v3;
      v6 = v40;
      v24 = v20;
      v25 = 0;
      v42 = v10;
      v26 = v23;
      v27 = a1[7];
      do
      {
        v45 = v26;
        v44 = v25;
        sub_11D33F0(v26, v21, v22[1]);
        v26 = v45;
        if ( !v6 )
        {
          v6 = v21;
          v25 = 1;
          goto LABEL_67;
        }
        v25 = v44;
        v28 = *(_QWORD *)(*(_QWORD *)(v6 + 72) + 80LL);
        if ( v28 )
        {
          v29 = v28 - 24;
          if ( v29 != v6 && v29 != v21 )
          {
            v30 = *(_DWORD *)(v27 + 32);
            v31 = (unsigned int)(*(_DWORD *)(v6 + 44) + 1);
            v28 = 0;
            if ( (unsigned int)v31 < v30 )
              goto LABEL_52;
LABEL_53:
            if ( v21 )
              goto LABEL_54;
            v32 = 0;
            v33 = 0;
            goto LABEL_55;
          }
          goto LABEL_62;
        }
        if ( v21 )
        {
          v31 = (unsigned int)(*(_DWORD *)(v6 + 44) + 1);
          v30 = *(_DWORD *)(v27 + 32);
          if ( v30 > (unsigned int)v31 )
          {
LABEL_52:
            v28 = *(_QWORD *)(*(_QWORD *)(v27 + 24) + 8 * v31);
            goto LABEL_53;
          }
LABEL_54:
          v32 = (unsigned int)(*(_DWORD *)(v21 + 44) + 1);
          v33 = *(_DWORD *)(v21 + 44) + 1;
LABEL_55:
          v34 = 0;
          if ( v33 < v30 )
            v34 = *(_QWORD *)(*(_QWORD *)(v27 + 24) + 8 * v32);
          while ( v28 != v34 )
          {
            if ( *(_DWORD *)(v28 + 16) < *(_DWORD *)(v34 + 16) )
            {
              v35 = v28;
              v28 = v34;
              v34 = v35;
            }
            v28 = *(_QWORD *)(v28 + 8);
          }
          v29 = *(_QWORD *)v34;
LABEL_62:
          if ( v29 != v6 )
            v25 = 0;
          if ( v21 == v29 )
            v25 = 1;
          goto LABEL_66;
        }
        v29 = 0;
        v25 = 1;
LABEL_66:
        v6 = v29;
LABEL_67:
        v22 += 4;
        if ( v22 == v24 )
          break;
        while ( 1 )
        {
          v21 = *v22;
          if ( *v22 != -8192 && v21 != -4096 )
            break;
          v22 += 4;
          if ( v24 == v22 )
            goto LABEL_71;
        }
      }
      while ( v22 != v24 );
LABEL_71:
      v10 = v42;
      v3 = v45;
      if ( !v25 )
LABEL_10:
        sub_11D33F0(v3, v6, v38);
      v7 = sub_11D7E40(v3, v40);
      if ( *(_QWORD *)(v10 - 96) )
      {
        v8 = *(_QWORD *)(v10 - 88);
        **(_QWORD **)(v10 - 80) = v8;
        if ( v8 )
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v10 - 80);
      }
      *(_QWORD *)(v10 - 96) = v7;
      if ( v7 )
      {
        v9 = *(_QWORD *)(v7 + 16);
        *(_QWORD *)(v10 - 88) = v9;
        if ( v9 )
          *(_QWORD *)(v9 + 16) = v10 - 88;
        *(_QWORD *)(v10 - 80) = v7 + 16;
        *(_QWORD *)(v7 + 16) = v10 - 96;
      }
LABEL_18:
      ++v41;
    }
    while ( (__int64 *)v37 != v41 );
  }
  return sub_11D2C20(v3);
}
