// Function: sub_2A6C900
// Address: 0x2a6c900
//
void __fastcall sub_2A6C900(__int64 a1, __int64 a2)
{
  unsigned __int8 *v4; // rax
  __int64 *v5; // rbx
  _BYTE *v6; // rdi
  unsigned __int8 *v7; // rax
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 v12; // rax
  _BYTE *v13; // r9
  __int64 v14; // rax
  unsigned int v15; // r14d
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  int v20; // eax
  int v21; // r10d
  unsigned __int64 v22; // [rsp+8h] [rbp-98h]
  unsigned __int8 v23[48]; // [rsp+10h] [rbp-90h] BYREF
  __int64 v24[12]; // [rsp+40h] [rbp-60h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 15
    || (*(_BYTE *)(a2 + 2) & 1) != 0
    || (v24[0] = a2, *(_BYTE *)sub_2A686D0(a1 + 136, v24) == 6) )
  {
    sub_2A6A450(a1, a2);
    return;
  }
  v4 = (unsigned __int8 *)sub_2A68BC0(a1, *(unsigned __int8 **)(a2 - 32));
  sub_22C05A0((__int64)v23, v4);
  if ( v23[0] <= 1u )
    goto LABEL_13;
  v24[0] = a2;
  v5 = sub_2A686D0(a1 + 136, v24);
  if ( !(unsigned __int8)sub_2A62D90((__int64)v23) )
    goto LABEL_11;
  v6 = (_BYTE *)sub_2A637C0(a1, (__int64)v23, *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL));
  if ( *v6 == 20 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
      v14 = **(_QWORD **)(v14 + 16);
    v15 = *(_DWORD *)(v14 + 8);
    v16 = sub_B43CB0(a2);
    if ( sub_B2F070(v16, v15 >> 8) )
      sub_2A634B0(a1, (unsigned __int8 *)v5, a2, v17, v18, v19);
    goto LABEL_13;
  }
  if ( *v6 == 3 )
  {
    if ( *(_DWORD *)(a1 + 216) )
    {
      v9 = *(unsigned int *)(a1 + 224);
      v10 = *(_QWORD *)(a1 + 208);
      if ( (_DWORD)v9 )
      {
        v11 = (v9 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v12 = v10 + 48LL * v11;
        v13 = *(_BYTE **)v12;
        if ( v6 == *(_BYTE **)v12 )
        {
LABEL_16:
          if ( v12 != v10 + 48 * v9 )
          {
            v22 = ((unsigned __int64)(unsigned int)qword_500BEC8 << 32) | 0x100;
            sub_22C05A0((__int64)v24, (unsigned __int8 *)(v12 + 8));
            sub_2A639B0(a1, v5, a2, (__int64)v24, v22);
            sub_22C0090((unsigned __int8 *)v24);
            goto LABEL_13;
          }
        }
        else
        {
          v20 = 1;
          while ( v13 != (_BYTE *)-4096LL )
          {
            v21 = v20 + 1;
            v11 = (v9 - 1) & (v20 + v11);
            v12 = v10 + 48LL * v11;
            v13 = *(_BYTE **)v12;
            if ( v6 == *(_BYTE **)v12 )
              goto LABEL_16;
            v20 = v21;
          }
        }
      }
    }
  }
  v7 = (unsigned __int8 *)sub_9718F0((__int64)v6, *(_QWORD *)(a2 + 8), *(_BYTE **)a1);
  if ( !v7 )
  {
LABEL_11:
    sub_2A62B80(v24, (unsigned __int8 *)a2);
    sub_2A689D0(a1, a2, (unsigned __int8 *)v24, 0x100000000LL);
    sub_22C0090((unsigned __int8 *)v24);
    sub_22C0090(v23);
    return;
  }
  sub_2A63320(a1, (__int64)v5, a2, v7, 0, v8);
LABEL_13:
  sub_22C0090(v23);
}
