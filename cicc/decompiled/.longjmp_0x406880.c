// Function: .longjmp
// Address: 0x406880
//
// attributes: thunk
void __noreturn longjmp(struct __jmp_buf_tag env[1], int val)
{
  longjmp(env, val);
}
