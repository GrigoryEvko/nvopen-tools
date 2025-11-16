// Function: sub_2F8B920
// Address: 0x2f8b920
//
void __fastcall sub_2F8B920(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 v12; // r11
  __int64 v13; // r14
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r15
  char v19; // dl
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  char *v26; // rdi
  __int64 v28; // [rsp+8h] [rbp-A8h]
  __int64 v29; // [rsp+10h] [rbp-A0h]
  __int64 v30; // [rsp+18h] [rbp-98h]
  __int64 v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+18h] [rbp-98h]
  __int64 v33; // [rsp+20h] [rbp-90h]
  int v34; // [rsp+28h] [rbp-88h]
  char v35; // [rsp+2Ch] [rbp-84h]
  char *v36[2]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE v37[48]; // [rsp+40h] [rbp-70h] BYREF
  int v38; // [rsp+70h] [rbp-40h]

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a5;
      if ( a4 + a5 == 2 )
      {
        v15 = a1;
        v13 = a2;
LABEL_12:
        if ( *(_DWORD *)(v13 + 8) > *(_DWORD *)(v15 + 8) )
        {
          v17 = *(_QWORD *)v15;
          v34 = *(_DWORD *)(v15 + 8);
          v18 = v15 + 16;
          v19 = *(_BYTE *)(v15 + 12);
          v36[0] = v37;
          v36[1] = (char *)0x600000000LL;
          v35 = v19;
          v20 = *(unsigned int *)(v15 + 24);
          v33 = v17;
          if ( (_DWORD)v20 )
          {
            v32 = v15;
            sub_2F8ABB0((__int64)v36, (char **)(v15 + 16), v20, v17, a5, v7);
            v15 = v32;
          }
          v31 = v15;
          v38 = *(_DWORD *)(v15 + 80);
          *(_QWORD *)v15 = *(_QWORD *)v13;
          *(_DWORD *)(v15 + 8) = *(_DWORD *)(v13 + 8);
          v21 = *(unsigned __int8 *)(v13 + 12);
          *(_BYTE *)(v15 + 12) = v21;
          sub_2F8ABB0(v18, (char **)(v13 + 16), v21, v17, a5, v7);
          v22 = *(unsigned int *)(v13 + 80);
          *(_DWORD *)(v31 + 80) = v22;
          *(_QWORD *)v13 = v33;
          *(_DWORD *)(v13 + 8) = v34;
          *(_BYTE *)(v13 + 12) = v35;
          sub_2F8ABB0(v13 + 16, v36, v22, v23, v24, v25);
          v26 = v36[0];
          *(_DWORD *)(v13 + 80) = v38;
          if ( v26 != v37 )
            _libc_free((unsigned __int64)v26);
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v13 = sub_2F8AD70(v7, a3, v6 + 88 * (v5 / 2));
        v14 = 0x2E8BA2E8BA2E8BA3LL * ((v13 - v11) >> 3);
        while ( 1 )
        {
          v30 = v14;
          v29 = v12;
          v28 = sub_2F8B420(v12, v11, v13, v10, v14, v11);
          sub_2F8B920(v6, v29, v28, v9, v30);
          a5 = v30;
          v8 -= v30;
          v5 -= v9;
          if ( !v5 )
            break;
          v15 = v28;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v6 = v28;
          v7 = v13;
          if ( v5 > v8 )
            goto LABEL_5;
LABEL_10:
          v13 = v7 + 88 * (v8 / 2);
          v16 = sub_2F8AD10(v6, v7, v13);
          v14 = v8 / 2;
          v12 = v16;
          v10 = (v16 - v6) >> 3;
          v9 = 0x2E8BA2E8BA2E8BA3LL * v10;
        }
      }
    }
  }
}
